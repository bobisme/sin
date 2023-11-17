pub mod logging;
pub mod sound;
pub mod ui;

use std::rc::Rc;

use color_eyre::Result;
use cpal::traits::{DeviceTrait, HostTrait};

use logging::prelude::*;
use sound::sig;

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
        cpal::SampleFormat::F32 => {
            let s = sig::<f32>(&config.clone().into());
            if let Err(e) = sound::play(Rc::new(device), &config.into(), s) {
                error!(?e, "play failed");
            };
        }
        cpal::SampleFormat::I16 => {
            let s = sig::<i16>(&config.clone().into());
            if let Err(e) = sound::play(Rc::new(device), &config.into(), s) {
                error!(?e, "play failed");
            };
        }
        cpal::SampleFormat::U16 => {
            let s = sig::<u16>(&config.clone().into());
            if let Err(e) = sound::play(Rc::new(device), &config.into(), s) {
                error!(?e, "play failed");
            };
        }
        _ => todo!(),
    };
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
