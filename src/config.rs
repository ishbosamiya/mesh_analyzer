use std::fmt::Display;

use quick_renderer::egui;

use crate::draw_ui::DrawUI;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Element {
    Node,
    Vert,
    Edge,
    Face,
}

impl Element {
    pub fn all() -> impl Iterator<Item = Self> {
        [Element::Node, Element::Vert, Element::Edge, Element::Face]
            .iter()
            .copied()
    }
}

impl Display for Element {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Element::Node => write!(f, "Node"),
            Element::Vert => write!(f, "Vert"),
            Element::Edge => write!(f, "Edge"),
            Element::Face => write!(f, "Face"),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Config {
    element: Element,
    element_index: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            element: Element::Node,
            element_index: 0,
        }
    }
}

impl DrawUI for Config {
    fn draw_ui(&self, _ui: &mut egui::Ui) {}

    fn draw_ui_edit(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("Element Type")
            .selected_text(format!("{}", self.element))
            .show_ui(ui, |ui| {
                Element::all().for_each(|element| {
                    ui.selectable_value(&mut self.element, element, format!("{}", element));
                });
            });

        ui.add(egui::Slider::new(&mut self.element_index, 0..=usize::MAX).text("Element Index"));
    }
}
