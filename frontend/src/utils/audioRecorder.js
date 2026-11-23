function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

export function encodeWavFromFloat32(samples, sampleRate) {
  const numSamples = samples.length;
  const dataSize = numSamples * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // PCM header size
  view.setUint16(20, 1, true); // PCM linear
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function mergeFloat32Chunks(chunks) {
  const totalLength = chunks.reduce((acc, arr) => acc + arr.length, 0);
  const merged = new Float32Array(totalLength);
  let pos = 0;
  for (let i = 0; i < chunks.length; i += 1) {
    merged.set(chunks[i], pos);
    pos += chunks[i].length;
  }
  return merged;
}

function mapMicrophoneError(err) {
  const name = err?.name || '';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return 'Permiso de micrófono denegado. Habilita el acceso al micrófono en tu navegador e inténtalo de nuevo.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No se detectó ningún micrófono conectado en este equipo.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'El micrófono está siendo utilizado por otra aplicación.';
  }
  return err?.message || 'No fue posible iniciar la grabación desde el micrófono.';
}

export async function createMicrophoneRecorder({
  onLevel,
  onTick,
  maxDurationSec = 15,
  onAutoStop,
} = {}) {
  if (!navigator?.mediaDevices?.getUserMedia) {
    throw new Error(
      'Este navegador no soporta captura de audio por micrófono (requiere localhost o HTTPS).'
    );
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (err) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (fallbackErr) {
      throw new Error(mapMicrophoneError(fallbackErr || err));
    }
  }

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) {
    stream.getTracks().forEach((t) => t.stop());
    throw new Error('Web Audio API no está disponible en este navegador.');
  }

  const audioCtx = new AudioCtx();
  if (audioCtx.state === 'suspended') {
    await audioCtx.resume();
  }

  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);
  const muteGain = audioCtx.createGain();
  muteGain.gain.value = 0;

  const chunks = [];
  const sampleRate = audioCtx.sampleRate;
  const startTimeMs = performance.now();
  let stopped = false;
  let rafId = null;
  let tickTimer = null;
  let autoStopTriggered = false;

  processor.onaudioprocess = (event) => {
    if (stopped) return;
    const input = event.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(input));
  };

  source.connect(analyser);
  source.connect(processor);
  processor.connect(muteGain);
  muteGain.connect(audioCtx.destination);

  const timeData = new Float32Array(analyser.fftSize);
  function updateLevel() {
    if (stopped) return;
    analyser.getFloatTimeDomainData(timeData);
    let sumSq = 0;
    for (let i = 0; i < timeData.length; i += 1) {
      sumSq += timeData[i] * timeData[i];
    }
    const rms = Math.sqrt(sumSq / timeData.length);
    const normalized = Math.min(1, rms * 5.0);
    if (onLevel) onLevel(normalized);
    rafId = requestAnimationFrame(updateLevel);
  }
  rafId = requestAnimationFrame(updateLevel);

  tickTimer = setInterval(() => {
    if (stopped) return;
    const elapsed = (performance.now() - startTimeMs) / 1000;
    if (onTick) onTick(Math.min(maxDurationSec, elapsed));
    if (elapsed >= maxDurationSec && !autoStopTriggered) {
      autoStopTriggered = true;
      if (onAutoStop) onAutoStop();
    }
  }, 200);

  async function cleanup() {
    if (stopped) return;
    stopped = true;
    if (rafId !== null) cancelAnimationFrame(rafId);
    if (tickTimer !== null) clearInterval(tickTimer);
    try {
      processor.disconnect();
      analyser.disconnect();
      source.disconnect();
      muteGain.disconnect();
    } catch {
      // Ignorar si ya fue desconectado
    }
    stream.getTracks().forEach((track) => track.stop());
    if (audioCtx.state !== 'closed') {
      await audioCtx.close();
    }
  }

  return {
    async stop() {
      await cleanup();
      const merged = mergeFloat32Chunks(chunks);
      const durationSec = merged.length / sampleRate;
      if (durationSec < 0.5) {
        throw new Error(
          'La grabación fue demasiado corta (menos de 0.5 s). Graba al menos 1 segundo de sonido.'
        );
      }
      const wavBlob = encodeWavFromFloat32(merged, sampleRate);
      const file = new File([wavBlob], 'grabacion_microfono.wav', {
        type: 'audio/wav',
      });
      const url = URL.createObjectURL(wavBlob);
      return {
        file,
        url,
        durationSec: Number(durationSec.toFixed(2)),
      };
    },
    async cancel() {
      await cleanup();
    },
  };
}
