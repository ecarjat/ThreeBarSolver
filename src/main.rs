use eframe::egui;

mod app;
mod config;
mod geometry;
mod linkage;
mod optimization;
mod types;

use app::state::AppState;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("3-Bar Linkage Designer"),
        ..Default::default()
    };

    eframe::run_native(
        "3-Bar Linkage Designer",
        options,
        Box::new(|_cc| Ok(Box::new(ThreeBarApp::new()))),
    )
}

struct ThreeBarApp {
    state: AppState,
}

impl ThreeBarApp {
    fn new() -> Self {
        Self {
            state: AppState::new(),
        }
    }
}

impl eframe::App for ThreeBarApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Side panel with controls
        egui::SidePanel::left("config_panel")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                app::sidebar::render_sidebar(ui, &mut self.state);
            });

        // Central panel with plot and results
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Left: Plot
                ui.vertical(|ui| {
                    ui.heading("Mechanism Pose");
                    app::plot::render_linkage_plot(ui, &mut self.state);
                });

                ui.separator();

                // Right: Results
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.vertical(|ui| {
                        app::results::render_results(ui, &mut self.state);
                    });
                });
            });
        });
    }
}
