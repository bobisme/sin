use std::rc::Rc;
use std::sync::Arc;

use color_eyre::Result;
use cpal::traits::{DeviceTrait, HostTrait};
use eframe::egui;
use eframe::epaint::{ColorImage, TextureHandle};
use itertools::Itertools;
use plotters::backend::{PixelFormat, RGBPixel};
use plotters::prelude::*;

use crate::logging::prelude::*;
use crate::sound;

const fn rgbcolor_from(color: egui::Color32) -> RGBColor {
    RGBColor(color.r(), color.g(), color.b())
}

const WIDTH: usize = 1200;
const HEIGHT: usize = 600;
const WAVEFORM: RGBColor = rgbcolor_from(catppuccin_egui::MOCHA.sapphire);
const BG_COLOR: RGBColor = rgbcolor_from(catppuccin_egui::MOCHA.base);
const TEXT: RGBColor = rgbcolor_from(catppuccin_egui::MOCHA.text);
const AXIS: RGBColor = rgbcolor_from(catppuccin_egui::MOCHA.overlay0);

pub enum WavPlotData {
    Vec(Vec<f32>),
    // Ring(dasp::ring_buffer::Bounded<[f32; 512]>),
}

impl WavPlotData {
    pub fn len(&self) -> usize {
        match self {
            WavPlotData::Vec(v) => v.len(),
            // WavPlotData::Ring(r) => r.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct WavPlot {
    pub plot_buf: Rc<[u8]>,
    pub plot_data_changed: bool,
    pub texture: Option<TextureHandle>,
    pub samples_per_sec: usize,
    data: Option<WavPlotData>,
}

impl WavPlot {
    pub fn new() -> Self {
        Self {
            plot_buf: Rc::new([0u8; WIDTH * HEIGHT * <RGBPixel as PixelFormat>::PIXEL_SIZE]),
            plot_data_changed: true,
            texture: None,
            samples_per_sec: 44_100,
            data: Default::default(),
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        if self.plot_data_changed {
            self.plot().unwrap();
            self.plot_data_changed = false;
            let img = egui::ImageData::Color(Arc::new(ColorImage::from_rgb(
                [WIDTH, HEIGHT],
                &self.plot_buf,
            )));
            self.texture = Some(ui.ctx().load_texture("plot.png", img, Default::default()));
        }
        if let Some(tex) = self.texture.as_ref() {
            ui.image((tex.id(), tex.size_vec2()));
        }
    }

    pub fn data_len(&self) -> usize {
        match self.data.as_ref() {
            Some(WavPlotData::Vec(v)) => v.len(),
            // Some(WavPlotData::Ring(r)) => r.len(),
            None => 0,
        }
    }

    pub fn plot(&mut self) -> Result<()> {
        let data_len = self.data_len();
        let root_area = BitMapBackend::with_buffer(
            Rc::get_mut(&mut self.plot_buf).unwrap(),
            (WIDTH as u32, HEIGHT as u32),
        )
        .into_drawing_area();

        root_area.fill(&BG_COLOR)?;

        // let root_area = root_area.titled("Image Title", ("sans-serif", 60))?;

        // let (upper, lower) = root_area.split_vertically(512);
        let upper = &root_area;

        // let x_axis = (0..self.data.len()).step(44_100 / 4);

        let mut cc = ChartBuilder::on(upper)
            .margin(5)
            .set_all_label_area_size(50)
            // .caption("Sine and Cosine", ("sans-serif", 40))
            .build_cartesian_2d(0.0f32..(data_len as f32), -1.0f32..1.0f32)?;

        cc.configure_mesh()
            .x_labels(20)
            .y_labels(10)
            .disable_mesh()
            .x_label_formatter(&|v| format!("{:.0}k", v / 1000.0))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .label_style(("sans-serif", 16, &TEXT))
            .axis_style(ShapeStyle {
                color: AXIS.mix(1.0),
                filled: false,
                stroke_width: 1,
            })
            .bold_line_style(ShapeStyle {
                color: AXIS.mix(1.0),
                filled: false,
                stroke_width: 2,
            })
            .draw()?;

        let data = match self.data.as_ref() {
            Some(WavPlotData::Vec(x)) => x.clone(),
            // Some(WavPlotData::Ring(x)) => {
            //     let (a, b) = x.slices();
            //     let mut v = Vec::with_capacity(a.len() + b.len());
            //     v.extend_from_slice(a);
            //     v.extend_from_slice(b);
            //     v
            // }
            None => vec![],
        };
        cc.draw_series(AreaSeries::new(
            data.iter().enumerate().map(|(i, x)| (i as f32, *x)),
            0.0,
            WAVEFORM,
        ))?;
        cc.draw_series(AreaSeries::new(
            data.iter().enumerate().map(|(i, x)| (i as f32, -*x)),
            0.0,
            WAVEFORM,
        ))?;
        // .label("Sine")
        // .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        // To avoid the IO failure being ignored silently, we manually call the present function
        root_area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
        Ok(())
    }

    pub fn set_data(&mut self, data: WavPlotData) {
        self.data = Some(data);
        self.plot_data_changed = true;
    }
}

impl Default for WavPlot {
    fn default() -> Self {
        Self::new()
    }
}

pub struct App {
    pub wav_plot: WavPlot,
    player: sound::Player,
}

impl App {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("failed to find a default output device");
        let config = device.default_output_config()?;
        let player = sound::Player::new(device, config);
        Ok(Self {
            wav_plot: Default::default(),
            player,
        })
    }

    pub fn run(self) -> Result<(), eframe::Error> {
        // env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
        let options = eframe::NativeOptions {
            initial_window_size: Some(egui::vec2(320.0, 240.0)),
            ..Default::default()
        };
        eframe::run_native(
            "My egui App",
            options,
            Box::new(|cc| {
                // This gives us image support:
                egui_extras::install_image_loaders(&cc.egui_ctx);
                cc.egui_ctx
                    .add_bytes_loader(Arc::new(egui::load::DefaultBytesLoader::default()));
                Box::new(self)
            }),
        )
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        catppuccin_egui::set_theme(ctx, catppuccin_egui::MOCHA);
        let sample_rate = self.player.sample_rate();
        self.wav_plot.samples_per_sec = sample_rate as usize;

        // self.wav_plot
        //     .set_data(WavPlotData::Vec(env.until_exhausted().collect::<Vec<_>>()));
        // let wav_data = self
        //     .player
        //     .ring_buf
        //     .read_arc()
        //     .iter()
        //     .cloned()
        //     .collect_vec();
        // self.wav_plot.set_data(WavPlotData::Vec(wav_data));

        egui::CentralPanel::default().show(ctx, |ui| {
            // ui.heading("My egui Application");
            if ui.button("play").clicked() {
                info!("clicked play");
                self.player.play()
            }
            self.wav_plot.show(ui);
        });
    }
}
