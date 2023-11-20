pub mod logging;
pub mod sound;
pub mod ui;

use std::rc::Rc;

use color_eyre::Result;
use cpal::{
    traits::{DeviceTrait, HostTrait},
    StreamConfig,
};

use dasp_graph::{node::Sum, BoxedNode, BoxedNodeSend, Node, NodeData};

use logging::prelude::*;

const MAX_NODES: usize = 1024;
const MAX_EDGES: usize = 1024;

struct SomeNode;

impl Node for SomeNode {
    fn process(&mut self, inputs: &[dasp_graph::Input], output: &mut [dasp_graph::Buffer]) {}
}

fn do_it<T>(device: cpal::Device, config: &StreamConfig) -> Result<()>
where
    T: cpal::FromSample<f32> + cpal::SizedSample + Send + 'static,
{
    info!("build graph");
    let mut graph = sound::Graph::with_capacity(MAX_NODES, MAX_EDGES);
    let mut processor = sound::Processor::with_capacity(MAX_NODES);
    let player = sound::PlayerNode::<f32>::new(device, config.clone());
    let signal = sound::sig::<T>(config.sample_rate.0);
    let signal = Box::new(sound::SigNode::<T, _>::new(config.sample_rate.0, signal));
    let sum_id = graph.add_node(NodeData::new1(BoxedNode(Box::new(Sum))));
    // let player_id = graph.add_node(NodeData::new1(BoxedNodeSend(Box::new(player))));
    let player_id = graph.add_node(NodeData::new1(BoxedNode(Box::new(player))));
    let signal_id = graph.add_node(NodeData::new1(BoxedNode(signal)));
    let _ = graph.add_edge(signal_id, sum_id, ());
    let _ = graph.add_edge(sum_id, player_id, ());
    // processor.process(&mut graph, signal_id);
    loop {
        processor.process(&mut graph, player_id);
    }
    // info!("sleep");
    // std::thread::sleep(std::time::Duration::from_secs(10));
    // if let Err(e) = sound::play(Rc::new(device), config, signal) {
    //     error!(?e, "play failed");
    // };
    Ok(())
}

fn main() -> Result<()> {
    logging::setup();

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("failed to find a default output device");
    // dbg!("default", device.name()?);
    // return Ok(());
    let config = device.default_output_config()?;
    // let mut player = sound::Player::new(device, config);

    // let s = sig::<f32>(&config.clone().into());
    match config.sample_format() {
        cpal::SampleFormat::F32 => do_it::<f32>(device, &config.clone().into()),
        cpal::SampleFormat::I16 => do_it::<i16>(device, &config.clone().into()),
        cpal::SampleFormat::U16 => do_it::<u16>(device, &config.clone().into()),
        _ => todo!(),
    }
    .unwrap();
    // sound::play(Rc::new(device), &config.into(), s).unwrap();
    // player.play();
    // let buf_data = [0f32; 4096];
    // let rbuf: Arc<RwLock<dasp::ring_buffer::Fixed<[f32; 4096]>>> =
    //     Arc::new(RwLock::new(dasp::ring_buffer::Fixed::from(buf_data)));

    // let s = match config.sample_format() {
    //     cpal::SampleFormat::F32 => sig::<f32>(&device, &config.into())?,
    //     cpal::SampleFormat::I16 => sig::<i16>(&device, &config.into())?,
    //     cpal::SampleFormat::U16 => sig::<u16>(&device, &config.into())?,
    //     _ => todo!(),
    // };

    // let app = crate::ui::App::new()?;
    // app.run().unwrap();

    Ok(())
}
