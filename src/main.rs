pub mod logging;
pub mod sound;
pub mod ui;

use std::sync::mpsc;

use color_eyre::Result;
use cpal::{
    traits::{DeviceTrait, HostTrait},
    StreamConfig,
};

use dasp_graph::{node::Sum, BoxedNodeSend, NodeData};

use logging::prelude::*;
use ringbuf::HeapRb;

const MAX_NODES: usize = 1024;
const MAX_EDGES: usize = 1024;
pub type Graph = petgraph::Graph<NodeData<BoxedNodeSend>, (), petgraph::Directed, u32>;
pub type Processor = dasp_graph::Processor<Graph>;

fn do_it<T>(device: cpal::Device, config: &StreamConfig) -> Result<()>
where
    T: cpal::FromSample<f32> + cpal::SizedSample + Send + 'static,
{
    info!("build graph");

    let env_rb = HeapRb::new(16384000);
    let (env_prod, env_cons) = env_rb.split();

    let mut graph = Graph::with_capacity(MAX_NODES, MAX_EDGES);
    let mut processor = Processor::with_capacity(MAX_NODES);

    let sum_id = graph.add_node(NodeData::new1(BoxedNodeSend(Box::new(Sum))));

    let player_rb = HeapRb::new(16384);
    let (player_prod, player_cons) = player_rb.split();
    let player_node = Box::new(sound::RingWriteNode::new(player_prod));
    let player_id = graph.add_node(NodeData::new1(BoxedNodeSend(Box::new(player_node))));
    let _stream = sound::player_stream::<f32>(device, config.clone(), player_cons);

    // let signal = sound::sig::<T>(config.sample_rate.0);
    let (sig_send, sig_recv) = mpsc::sync_channel(16);
    let signal_node = Box::new(sound::SigNode::<T>::new(config.sample_rate.0, sig_recv));
    let signal_id = graph.add_node(NodeData::new1(BoxedNodeSend(signal_node)));

    let env_node = Box::<sound::EnvelopeNode<_>>::default();
    let env_id = graph.add_node(NodeData::new1(BoxedNodeSend(env_node)));
    let env_write_node = Box::new(sound::RingWriteNode::new(env_prod));
    let env_write_node_id = graph.add_node(NodeData::new1(BoxedNodeSend(env_write_node)));

    let _ = graph.add_edge(signal_id, sum_id, ());
    let _ = graph.add_edge(sum_id, player_id, ());
    let _ = graph.add_edge(player_id, env_id, ());
    let _ = graph.add_edge(env_id, env_write_node_id, ());

    info!("graph built");
    let (send_kill, recv_kill) = std::sync::mpsc::channel();
    let audio_thread = std::thread::spawn(move || loop {
        match recv_kill.try_recv() {
            Ok(_) => break,
            Err(_) => {
                processor.process(&mut graph, env_write_node_id);
            }
        }
    });
    let app = crate::ui::App::new(env_cons, sig_send);
    app.run().expect("ow, my hip!");
    send_kill.send(()).unwrap();
    audio_thread.join().unwrap();
    info!("poop");
    Ok(())
}

fn main() -> Result<()> {
    logging::setup();

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("failed to find a default output device");
    let config = device.default_output_config()?;

    match config.sample_format() {
        cpal::SampleFormat::F32 => do_it::<f32>(device, &config.clone().into()),
        cpal::SampleFormat::I16 => do_it::<i16>(device, &config.clone().into()),
        cpal::SampleFormat::U16 => do_it::<u16>(device, &config.clone().into()),
        _ => todo!(),
    }
    .unwrap();

    Ok(())
}
