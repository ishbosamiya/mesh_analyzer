use quick_renderer::egui;

pub trait DrawUI {
    fn draw_ui(&self, ui: &mut egui::Ui);
    fn draw_ui_edit(&mut self, ui: &mut egui::Ui);
}
