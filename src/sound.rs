use color_eyre::Result;
use const_soft_float::soft_f64::SoftF64;
use cpal::{
    traits::{DeviceTrait, StreamTrait},
    Device, StreamConfig, SupportedStreamConfig,
};
use dasp::{
    // envelope,
    signal::{self},
    Sample,
    Signal,
};
// use dasp::signal::envelope::SignalEnvelope;
use parking_lot::RwLock;
use std::{
    fmt::Debug,
    sync::{mpsc, Arc},
};

use crate::logging::prelude::*;

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
    signal: impl Iterator<Item = f32>,
) where
    T: cpal::Sample + cpal::FromSample<f32>,
{
    let mut i = signal.into_iter();
    for frame in output.chunks_mut(channels) {
        let sample = match i.next() {
            None => {
                complete_tx.try_send(()).ok();
                0.0
            }
            Some(sample) => sample,
        };
        // let value: T = cpal::Sample::from::<f32>(&sample);
        let value: T = sample.to_sample();
        for sample in frame {
            *sample = value;
        }
    }
}

fn write_chan<T>(
    output: &mut [T],
    channels: usize,
    complete_tx: &mpsc::SyncSender<()>,
    chan: std::sync::mpsc::Receiver<T>,
) where
    T: cpal::Sample + cpal::FromSample<f32> + Default,
{
    for frame in output.chunks_mut(channels) {
        let sample = match chan.try_recv() {
            Ok(sample) => sample,
            Err(_) => {
                complete_tx.try_send(()).ok();
                Default::default()
            }
        };
        // let value: T = cpal::Sample::from::<f32>(&sample);
        let value: T = sample.to_sample();
        for sample in frame {
            *sample = value;
        }
    }
}

fn write_signal<S, F>(
    output: &mut [F],
    channels: usize,
    complete_tx: &mpsc::SyncSender<()>,
    signal: &mut S,
) where
    // F: cpal::Sample + cpal::FromSample<f32> + Default,
    S: dasp::Signal<Frame = F>,
    F: dasp::Frame + Default,
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

pub fn sig<S>(config: &cpal::StreamConfig) -> impl signal::Signal<Frame = f32>
where
    S: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    // Create a signal chain to play back 1 second of each oscillator at A4.
    let hz = signal::rate(config.sample_rate.0 as f64).const_hz(TUNING_PITCH);
    let sample_rate = config.sample_rate.0;
    let t = (sample_rate * 6 / 8) as usize;
    let rest = (sample_rate * 2 / 8) as usize;
    let pitch_groups = [69usize, 76, 81].into_iter().flat_map(move |i| {
        let hz = signal::rate(sample_rate as f64).const_hz(PITCHES[i]);
        hz.clone()
            .sine()
            .take(t)
            .chain(std::iter::repeat(0.0).take(rest))
            .chain(hz.clone().square().take(t))
            .chain(std::iter::repeat(0.0).take(rest))
            .chain(hz.clone().noise_simplex().take(t))
            .chain(std::iter::repeat(0.0).take(rest))
    });

    const VOLUME: f32 = 0.2;

    let synth = hz
        .clone()
        .sine()
        .take(t)
        .chain(pitch_groups)
        .chain(signal::noise(0).take(t))
        .map(|s| s.to_sample::<f32>() * VOLUME);

    signal::from_iter(synth)
}

// pub fn play<F>(device: &cpal::Device, config: &cpal::StreamConfig, input: Vec<f32>) -> Result<()>
pub fn play<F>(
    device: impl AsRef<cpal::Device>,
    config: &cpal::StreamConfig,
    mut input: impl 'static + signal::Signal<Frame = F> + Send,
) -> Result<()>
where
    F: cpal::SizedSample + cpal::FromSample<f32> + Default + Debug,
{
    // A channel for indicating when playback has completed.
    let (complete_tx, complete_rx) = mpsc::sync_channel(1);

    // Create and run the stream.
    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);
    let channels = config.channels as usize;
    let callback = move |output: &mut [F], _: &cpal::OutputCallbackInfo| {
        info!("inside callback");
        for out_frame in output.chunks_mut(channels) {
            if input.is_exhausted() {
                complete_tx.send(()).unwrap();
                break;
            }
            let sample = input.next();
            let value: F = sample.to_sample();
            for sample in out_frame.iter_mut() {
                *sample = value;
                debug!(s = ?value, "put sample");
            }
        }
    };
    let stream = device
        .as_ref()
        .build_output_stream(config, callback, err_fn, None)?;
    info!("play");
    stream.play()?;

    // Wait for playback to complete.
    complete_rx.recv().unwrap();
    info!("pause");
    stream.pause()?;

    Ok(())
}

pub struct Player {
    // TODO: make lock-free
    pub ring_buf: Arc<RwLock<dasp::ring_buffer::Fixed<[f32; 4096]>>>,
    config: SupportedStreamConfig,
}

impl Player {
    pub fn new(device: Device, config: SupportedStreamConfig) -> Self {
        // let rbuf = dasp::ring_buffer::Fixed::from([0f32; 4096]);
        let x = Self {
            ring_buf: Arc::new(RwLock::new([0f32; 4096].into())),
            config,
        };
        let arc_buf = Arc::clone(&x.ring_buf);
        std::thread::spawn(move || Self::run(device, arc_buf));
        x
    }

    pub fn run(device: Device, ring_buf: Arc<RwLock<dasp::ring_buffer::Fixed<[f32; 4096]>>>) {
        info!("Player::run");
        let config = match device.default_output_config() {
            Ok(config) => StreamConfig::from(config),
            Err(err) => {
                error!(?err, "could not get ouput config");
                return;
            }
        };
        let channels = config.channels as usize;
        let (complete_tx, complete_rx) = mpsc::sync_channel(1);
        // let rb = Arc::clone(&ring_buf);
        let stream = device.build_output_stream(
            &config,
            move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                info!("play loop started");
                loop {
                    // sync
                    let rb = ring_buf.read_arc();
                    let mut i = rb.iter();
                    for ouput_frame in output.chunks_mut(channels) {
                        let Some(sample) = i.next() else {
                            break;
                        };
                        debug!(sample, "got sample");
                        for out_sample in ouput_frame {
                            *out_sample = sample.to_sample();
                        }
                    }
                    // std::thread::yield_now();
                }
                complete_tx.send(()).unwrap();
            },
            |err| eprintln!("an error occurred on stream: {}", err),
            None,
        );
        let Ok(stream) = stream else {
            error!(err = ?stream.err(), "could not build output stream");
            return;
        };
        info!("playing stream");
        if let Err(err) = stream.play() {
            error!(?err, "could not play stream");
        }
        // Wait for playback to complete.
        complete_rx.recv().unwrap();
        if let Err(err) = stream.pause() {
            error!(?err, "could not pause stream");
        }
        info!("Player::run ended");
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate().0
    }

    pub fn play(&mut self) {
        let sample_rate = self.sample_rate();
        let t = sample_rate as usize * 2;
        let s = sig::<f32>(&self.config.clone().into());
        // let v: Vec<_> = s.take(t).collect();
        // let attack = f32::max(1.0, t as f32 / 1000.0);
        // let detector = dasp::envelope::Detector::peak(attack, attack);
        // let mut fork = s.fork(ring);
        // let (a, b) = fork.by_ref();
        // let env = signal::from_iter(v.iter().copied()).detect_envelope(detector);
        debug!("writing to ring buffer");
        {
            let mut buf = self.ring_buf.write_arc();
            s.take(t).for_each(|f| {
                buf.push(f);
            });
            debug!("wrote");
        }
        // if let Some(buf) = Arc::get_mut(&mut self.ring_buf) {
        //     env.take(t).for_each(|f| {
        //         buf.push(f);
        //     });
        // } else {
        //     debug!("could not write");
        // }

        // let config = self.config.clone();
        // let s: Vec<_> = v.iter().map(|x| x * 0.3).take(t).collect();
    }
}
