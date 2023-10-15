use color_eyre::Result;
use const_soft_float::soft_f64::SoftF64;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use dasp::{
    envelope,
    signal::{self, envelope::SignalEnvelope},
    Sample, Signal,
};
use std::sync::mpsc;

// (2 ^ 1/12)
#[allow(clippy::excessive_precision)]
const ROOT_F64: f64 = 1.0594630943592952645618252949463417007792043174941856285592084314;
const TUNING_PITCH: f64 = 440.0;
const TUNING_IDX: usize = 69;

const PITCHES: [f64; 128] = {
    let mut pitches = [0f64; 128];
    let mut i = 128;
    while i > 0 {
        i -= 1;
        let rel = i as i32 - TUNING_IDX as i32;
        pitches[i] = SoftF64(TUNING_PITCH)
            .mul(SoftF64(ROOT_F64).powi(rel))
            .to_f64();
    }
    pitches
};

// compile-time tests
const _: () = {
    // octaves
    assert!(PITCHES[69 - 12] as u64 == 220);
    assert!(PITCHES[69] as u64 == 440);
    assert!(PITCHES[69 + 12] as u64 == 880);
    // spot checks
    // A0
    assert!(PITCHES[21] as u64 == 27);
    // C4
    assert!(PITCHES[60] as u64 == 261);
    // A7
    assert!(PITCHES[105] as u64 == 3520);
};

fn write_data<T>(
    output: &mut [T],
    channels: usize,
    complete_tx: &mpsc::SyncSender<()>,
    signal: &mut dyn Iterator<Item = f32>,
) where
    T: cpal::Sample + cpal::FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let sample = match signal.next() {
            None => {
                complete_tx.try_send(()).ok();
                0.0
            }
            Some(sample) => sample,
        };
        // let value: T = cpal::Sample::from::<f32>(&sample);
        let value: T = sample.to_sample();
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}

fn write_signal<T, S>(
    output: &mut [T],
    channels: usize,
    complete_tx: &mpsc::SyncSender<()>,
    signal: &mut S,
) where
    T: cpal::Sample + cpal::FromSample<f32> + Default,
    S: dasp::Signal<Frame = T>,
{
    for out_frame in output.chunks_mut(channels) {
        // let frame = signal.next();
        let frame = match signal.is_exhausted() {
            true => {
                complete_tx.try_send(()).ok();
                Default::default()
            }
            false => signal.next(),
        };
        // let value: T = cpal::Sample::from::<f32>(&sample);
        out_frame.iter_mut().for_each(|out| {
            *out = frame;
        });
    }
}

fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> Result<()>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    // Create a signal chain to play back 1 second of each oscillator at A4.
    let hz = signal::rate(config.sample_rate.0 as f64).const_hz(TUNING_PITCH);
    let sample_rate = config.sample_rate.0;
    let t = (sample_rate * 6 / 8) as usize;
    let rest = (sample_rate * 2 / 8) as usize;
    let pitch_groups = [69usize, 76, 81].into_iter().flat_map(move |i| {
        let hz = signal::rate(sample_rate as f64).const_hz(PITCHES[i]);
        hz.clone()
            .saw()
            .take(t)
            .chain(std::iter::repeat(0.0).take(rest))
            .chain(hz.clone().square().take(t))
            .chain(std::iter::repeat(0.0).take(rest))
            .chain(hz.clone().noise_simplex().take(t))
            .chain(std::iter::repeat(0.0).take(rest))
    });

    let volume: f32 = 0.2;

    let synth = hz
        .clone()
        .sine()
        .take(t)
        .chain(pitch_groups)
        .chain(signal::noise(0).take(t))
        .map(move |s| s.to_sample::<f32>() * volume.clamp(0.0, 0.9));

    let mut sig = signal::from_iter(synth);

    // let attack = sample_rate as f32 / 16.0;
    // let release = sample_rate as f32 / 2048.0;
    // let detector = dasp::envelope::Detector::peak(attack, release);
    // let mut env = sig.detect_envelope(detector);

    // A channel for indicating when playback has completed.
    let (complete_tx, complete_rx) = mpsc::sync_channel(1);

    // Create and run the stream.
    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);
    let channels = config.channels as usize;
    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            // write_data(data, channels, &complete_tx, &mut synth)
            write_signal(data, channels, &complete_tx, &mut sig)
        },
        err_fn,
        None,
    )?;
    stream.play()?;

    // Wait for playback to complete.
    complete_rx.recv().unwrap();
    stream.pause()?;

    Ok(())
}

fn main() -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("failed to find a default output device");
    let config = device.default_output_config()?;

    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into())?,
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into())?,
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into())?,
        _ => todo!(),
    }

    Ok(())
}
