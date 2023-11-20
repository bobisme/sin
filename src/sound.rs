use color_eyre::Result;
use const_soft_float::soft_f64::SoftF64;
use cpal::{
    traits::{DeviceTrait, StreamTrait},
    Device, Sample, StreamConfig,
};
use dasp::Frame;
use dasp_graph::{BoxedNode, Buffer, Node, NodeData};
use dasp_signal::Signal;
// use dasp::signal::envelope::SignalEnvelope;
use ringbuf::{HeapConsumer, HeapProducer};
use std::{fmt::Debug, marker::PhantomData};

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
pub type Graph = petgraph::Graph<NodeData<BoxedNode>, (), petgraph::Directed, u32>;
pub type Processor = dasp_graph::Processor<Graph>;

pub fn sig<S>(sample_rate: u32) -> dasp_signal::FromIterator<impl Iterator<Item = f32>>
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
    let synth = hz
        .clone()
        .sine()
        .take(t)
        .chain(pitch_groups)
        .chain(dasp_signal::noise(0).take(t))
        .map(|s| s.to_sample::<f32>() * VOLUME);

    dasp_signal::from_iter(synth)
}

pub struct SigNode<Sam, Sig> {
    signal: Sig,
    _mark: PhantomData<Sam>,
}

impl<Sam, Sig> SigNode<Sam, Sig>
where
    Sam: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
    Sig: Signal<Frame = f32>,
{
    pub fn new(signal: Sig) -> Self {
        Self {
            signal,
            _mark: PhantomData,
        }
    }
}

impl<Sam, Sig> Node for SigNode<Sam, Sig>
where
    Sam: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
    Sig: Signal<Frame = f32>,
{
    fn process(&mut self, _inputs: &[dasp_graph::Input], output: &mut [dasp_graph::Buffer]) {
        debug!("SigNode::process");
        // let channels = core::cmp::min(1, output.len());
        let channels = core::cmp::min(<Sig::Frame as Frame>::CHANNELS, output.len());
        for ix in 0..Buffer::LEN {
            if self.signal.is_exhausted() {
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
    let (complete_tx, complete_rx) = std::sync::mpsc::sync_channel(1);

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
        if cons.is_empty() {
            return;
        }
        let mut iter = cons.pop_iter();
        for out_frame in device_out.chunks_mut(channels) {
            let in_sample = match iter.next() {
                Some(sample) => sample.to_sample(),
                None => break,
            };
            for out_sample in out_frame.iter_mut() {
                *out_sample = in_sample;
            }
            debug!(s = ?in_sample, "PlayerNode callback: put sample");
        }
        debug!("leaving callback");
    };
    device
        .build_output_stream(&config, callback, err_fn, None)
        .unwrap()
}
