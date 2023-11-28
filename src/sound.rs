use color_eyre::Result;
use const_soft_float::soft_f64::SoftF64;
use cpal::{
    traits::{DeviceTrait, StreamTrait},
    Device, Sample, StreamConfig,
};
use dasp::{peak::PositiveHalfWave, Frame};
use dasp_envelope::detect::Peak;
use dasp_graph::{Buffer, Node};
use dasp_signal::Signal;
use itertools::Itertools;
// use dasp::signal::envelope::SignalEnvelope;
use ringbuf::{HeapConsumer, HeapProducer};
use std::{fmt::Debug, marker::PhantomData, sync::mpsc};

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

pub const MAX_NODES: usize = 1024;
pub const MAX_EDGES: usize = 1024;

pub fn sig_vec<S>(sample_rate: u32) -> Vec<f32>
where
    S: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    // Create a signal chain to play back 1 second of each oscillator at A4.
    let hz = dasp_signal::rate(sample_rate as f64).const_hz(TUNING_PITCH);
    let t = (sample_rate * 6 / 8) as usize;
    let rest = (sample_rate * 2 / 8) as usize;
    let pitch_groups = [69usize, 76, 81].into_iter().flat_map(move |i| {
        let hz = dasp_signal::rate(sample_rate as f64).const_hz(PITCHES[i]);
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
    hz.clone()
        .sine()
        .take(t)
        .chain(pitch_groups)
        .chain(dasp_signal::noise(0).take(t))
        .map(|s| s.to_sample::<f32>() * VOLUME)
        .collect_vec()
}

// pub fn sig<S, I>(sample_rate: u32) -> dasp_signal::FromIterator<impl Iterator<Item = f32>>
pub fn sig<S>(sample_rate: u32) -> dasp_signal::FromIterator<std::vec::IntoIter<f32>>
where
    S: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    dasp_signal::from_iter(sig_vec::<S>(sample_rate))
}

pub enum SigMsg {
    Play,
    Pause,
    Stop,
}

pub struct SigNode<Sam> {
    signal: dasp_signal::FromIterator<std::vec::IntoIter<f32>>,
    sample_rate: u32,
    play: bool,
    recv: mpsc::Receiver<SigMsg>,
    _mark: PhantomData<Sam>,
}

impl<Sam> SigNode<Sam>
where
    Sam: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    pub fn new(sample_rate: u32, recv: mpsc::Receiver<SigMsg>) -> Self {
        // let signal_data = sig_vec::<Sam>(sample_rate);
        let signal = sig::<Sam>(sample_rate);
        Self {
            signal,
            sample_rate,
            play: false,
            recv,
            _mark: PhantomData,
        }
    }

    pub fn handle(&mut self, msg: SigMsg) {
        match msg {
            // SigMsg::Play => self.play.store(true, Ordering::Release),
            // SigMsg::Pause | SigMsg::Stop => self.play.store(false, Ordering::Release),
            SigMsg::Play | SigMsg::Pause => {
                self.play = !self.play;
                if self.play && self.signal.is_exhausted() {
                    self.signal = sig::<Sam>(self.sample_rate);
                }
            }
            SigMsg::Stop => self.play = false,
        }
    }
}

impl<Sam> Node for SigNode<Sam>
where
    Sam: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    fn process(&mut self, _inputs: &[dasp_graph::Input], output: &mut [dasp_graph::Buffer]) {
        debug!("SigNode::process");
        match self.recv.try_recv() {
            Ok(msg) => self.handle(msg),
            Err(err) => match err {
                mpsc::TryRecvError::Empty => {}
                _ => error!(?err, "Signode::process failed getting message"),
            },
        }
        // let channels = core::cmp::min(<I::Item as Frame>::CHANNELS, output.len());
        let channels = core::cmp::min(1, output.len());
        // let channels = core::cmp::min(1, output.len());
        for ix in 0..Buffer::LEN {
            if self.signal.is_exhausted() || !self.play {
                for out_buffer in output.iter_mut() {
                    out_buffer.silence();
                }
            } else {
                let frame = self.signal.next();
                (0..channels).for_each(|ch| {
                    // Safe, as we verify the number of channels at the beginning of the function.
                    output[ch][ix] = unsafe { *frame.channel_unchecked(ch) };
                });
            }
        }
        debug!("SigNode::process END");
    }
}

// pub fn play<F>(device: &cpal::Device, config: &cpal::StreamConfig, input: Vec<f32>) -> Result<()>
pub fn play<F>(
    device: impl AsRef<cpal::Device>,
    config: &cpal::StreamConfig,
    mut input: impl 'static + dasp_signal::Signal<Frame = F> + Send,
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
        debug!("inside callback");
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
    debug!("play");
    stream.play()?;

    // Wait for playback to complete.
    complete_rx.recv().unwrap();
    debug!("pause");
    stream.pause()?;

    Ok(())
}

pub struct RingWriteNode {
    prod: HeapProducer<f32>,
    pass: dasp_graph::node::Pass,
}

impl Debug for RingWriteNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingWriteNode")
            .field("prod", &"...producer...")
            .finish()
    }
}

impl RingWriteNode {
    pub fn new(prod: HeapProducer<f32>) -> Self {
        Self {
            prod,
            pass: dasp_graph::node::Pass,
        }
    }
}

impl Node for RingWriteNode {
    #[tracing::instrument]
    fn process(&mut self, inputs: &[dasp_graph::Input], output: &mut [Buffer]) {
        let Some(input) = inputs.get(0) else {
            error!("no inputs");
            return;
        };
        let in_buffers = input.buffers();
        let Some(in_buf) = in_buffers.get(0) else {
            error!("no input buffer");
            return;
        };
        for sample in in_buf.iter() {
            while self.prod.is_full() {
                core::hint::spin_loop();
            }
            if let Err(val) = self.prod.push(*sample) {
                warn!(sample = val, "RingWriteNode: ring buffer full");
            };
        }
        self.pass.process(inputs, output)
    }
}

pub fn player_stream<Samp>(
    device: Device,
    config: StreamConfig,
    mut cons: HeapConsumer<f32>,
) -> cpal::Stream
where
    Samp: cpal::SizedSample + cpal::FromSample<f32> + Default + Debug + Send,
{
    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);
    let channels = config.channels as usize;
    let callback = move |device_out: &mut [Samp], _: &cpal::OutputCallbackInfo| {
        // loop {
        debug!("inside callback");
        for (out_frame, in_sample) in device_out.chunks_mut(channels).zip(cons.pop_iter()) {
            for out_sample in out_frame.iter_mut() {
                *out_sample = in_sample.to_sample();
            }
            debug!(s = ?in_sample, "player_stream callback: put sample");
        }
        debug!("leaving callback");
    };
    device
        .build_output_stream(&config, callback, err_fn, None)
        .unwrap()
}

pub struct EnvelopeNode<D>
where
    D: dasp_envelope::Detect<f32>,
{
    detector: dasp_envelope::Detector<f32, D>,
}

impl<D> EnvelopeNode<D>
where
    D: dasp_envelope::Detect<f32>,
{
    pub fn new(detector: dasp_envelope::Detector<f32, D>) -> Self {
        Self { detector }
    }
}

impl Default for EnvelopeNode<Peak<PositiveHalfWave>> {
    fn default() -> Self {
        let detector = dasp_envelope::Detector::peak_positive_half_wave(2.0, 2.0);
        Self { detector }
    }
}

use dasp_signal::envelope::SignalEnvelope;

impl<D> Node for EnvelopeNode<D>
where
    D: dasp_envelope::Detect<f32, Output = f32> + Clone,
{
    fn process(&mut self, inputs: &[dasp_graph::Input], output: &mut [Buffer]) {
        let input = match inputs.get(0) {
            None => return,
            Some(input) => input,
        };
        for (out_buf, in_buf) in output.iter_mut().zip(input.buffers()) {
            let sig = dasp_signal::from_iter(in_buf.iter().cloned());
            sig.detect_envelope(self.detector.clone())
                .take(Buffer::LEN)
                .enumerate()
                .for_each(|(i, sample)| out_buf[i] = sample);
        }
    }
}
