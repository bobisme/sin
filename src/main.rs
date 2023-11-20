pub mod logging;
pub mod sound;
pub mod ui;

use color_eyre::Result;
use cpal::{
    traits::{DeviceTrait, HostTrait},
    StreamConfig,
};

use dasp_graph::{node::Sum, BoxedNode, NodeData};

use logging::prelude::*;
use ringbuf::HeapRb;

const MAX_NODES: usize = 1024;
const MAX_EDGES: usize = 1024;

fn do_it<T>(device: cpal::Device, config: &StreamConfig) -> Result<()>
where
    T: cpal::FromSample<f32> + cpal::SizedSample + Send + 'static,
{
    info!("build graph");
    let mut graph = sound::Graph::with_capacity(MAX_NODES, MAX_EDGES);
    let mut processor = sound::Processor::with_capacity(MAX_NODES);
    let sum_id = graph.add_node(NodeData::new1(BoxedNode(Box::new(Sum))));
    let player_rb = HeapRb::new(16384);
    let (player_prod, player_cons) = player_rb.split();
    let player_node = Box::new(sound::RingWriteNode::new(player_prod));
    let player_id = graph.add_node(NodeData::new1(BoxedNode(Box::new(player_node))));
    let _stream = sound::player_stream::<f32>(device, config.clone(), player_cons);
    let signal = sound::sig::<T>(config.sample_rate.0);
    let signal_node = Box::new(sound::SigNode::<T, _>::new(signal));
    let signal_id = graph.add_node(NodeData::new1(BoxedNode(signal_node)));
    let _ = graph.add_edge(signal_id, sum_id, ());
    let _ = graph.add_edge(sum_id, player_id, ());
    loop {
        processor.process(&mut graph, player_id);
    }
    // info!("sleep");
    // std::thread::sleep(std::time::Duration::from_secs(10));
    // if let Err(e) = sound::play(Rc::new(device), config, signal) {
    //     error!(?e, "play failed");
    // };
    // Ok(())
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

    // let app = crate::ui::App::new()?;
    // app.run().unwrap();

    Ok(())
}
