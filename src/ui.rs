use std::sync::{mpsc, Arc};

use color_eyre::Result;
use eframe::egui;
use eframe::epaint::{ColorImage, TextureHandle};
use plotters::backend::{PixelFormat, RGBPixel};
use plotters::prelude::*;
use ringbuf::HeapConsumer;

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
    Ring(dasp_ring_buffer::Bounded<Vec<f32>>),
}

impl WavPlotData {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            WavPlotData::Vec(v) => v.len(),
            WavPlotData::Ring(r) => r.len(),
        }
    }

    pub fn push(&mut self, x: f32) {
        match self {
            WavPlotData::Vec(v) => {
                v.push(x);
            }
            WavPlotData::Ring(r) => {
                r.push(x);
            }
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct WavPlot<Buf> {
    pub plot_buf: Buf,
    pub plot_data_changed: bool,
    pub texture: Option<TextureHandle>,
    pub samples_per_sec: usize,
    data: Option<WavPlotData>,
}

impl<Buf> WavPlot<Buf>
where
    Buf: AsMut<[u8]> + AsRef<[u8]>,
{
    pub fn new(plot_buf: Buf) -> Self {
        info!("WavPlot::new");
        Self {
            plot_buf,
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
                self.plot_buf.as_ref(),
            )));
            self.texture = Some(ui.ctx().load_texture("plot.png", img, Default::default()));
        }
        if let Some(tex) = self.texture.as_ref() {
            ui.image((tex.id(), tex.size_vec2()));
        }
    }

    pub fn data_len(&self) -> usize {
        self.data.as_ref().map(WavPlotData::len).unwrap_or(0)
    }

    pub fn plot(&mut self) -> Result<()> {
        let data_len = self.data_len();
        let root_area =
            BitMapBackend::with_buffer(self.plot_buf.as_mut(), (WIDTH as u32, HEIGHT as u32))
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
            .build_cartesian_2d(0.0f32..(data_len as f32), -0.5f32..0.5f32)?;

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

        let data = match self.data.as_mut() {
            Some(WavPlotData::Vec(x)) => x.clone(),
            Some(WavPlotData::Ring(x)) => x.drain().collect::<Vec<_>>(),
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

impl Default for WavPlot<Vec<u8>> {
    fn default() -> Self {
        Self::new(Vec::with_capacity(
            WIDTH * HEIGHT * <RGBPixel as PixelFormat>::PIXEL_SIZE,
        ))
    }
}

// impl Default for WavPlot<Rc<[u8; WIDTH * HEIGHT * <RGBPixel as PixelFormat>::PIXEL_SIZE]>> {
//     fn default() -> Self {
//         let plot_buf = Rc::new([0u8; WIDTH * HEIGHT * <RGBPixel as PixelFormat>::PIXEL_SIZE]);
//         Self::new(plot_buf)
//     }
// }

pub struct App {
    pub wav_plot: WavPlot<Vec<u8>>,
    cons: HeapConsumer<f32>,
    // player: sound::Player,
    sig_send: mpsc::SyncSender<sound::SigMsg>,
}

impl App {
    pub fn new(cons: HeapConsumer<f32>, sig_send: mpsc::SyncSender<sound::SigMsg>) -> Self {
        info!("App::new");
        let wav_plot = WavPlot {
            // plot_buf: Rc::new([0u8; WIDTH * HEIGHT * <RGBPixel as PixelFormat>::PIXEL_SIZE]),
            plot_buf: vec![0u8; WIDTH * HEIGHT * <RGBPixel as PixelFormat>::PIXEL_SIZE],
            plot_data_changed: true,
            texture: None,
            samples_per_sec: 44_100,
            data: Some(WavPlotData::Ring(dasp_ring_buffer::Bounded::from(
                vec![0f32; 48_000],
            ))),
        };
        info!("App::new created wav_plot");
        Self {
            wav_plot,
            cons,
            sig_send,
        }
    }

    pub fn run(self) -> Result<(), eframe::Error> {
        info!("App::run");
        let options = eframe::NativeOptions {
            initial_window_size: Some(egui::vec2(1200.0, 700.0)),
            app_id: Some("sin".to_string()),
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

        self.wav_plot.plot_data_changed = true;
        for sample in self.cons.pop_iter() {
            if let Some(data) = self.wav_plot.data.as_mut() {
                data.push(sample)
            }
        }
        // let sample_rate = self.player.sample_rate();
        // self.wav_plot.samples_per_sec = sample_rate as usize;

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
                self.sig_send.send(sound::SigMsg::Play).unwrap();
            }
            self.wav_plot.show(ui);
        });
    }
}
