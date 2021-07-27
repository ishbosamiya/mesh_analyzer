use quick_renderer::egui;

pub trait DrawUI {
    type ExtraData;
    fn draw_ui(&self, extra_data: &Self::ExtraData, ui: &mut egui::Ui);
    fn draw_ui_edit(&mut self, extra_data: &Self::ExtraData, ui: &mut egui::Ui);
}
