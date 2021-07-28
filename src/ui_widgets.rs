use quick_renderer::egui;
use quick_renderer::glm;

use crate::math;

/// A 3D transform widget
#[derive(Debug, PartialEq, PartialOrd)]
pub struct Transform<'a> {
    transform: &'a mut math::Transform,
}

impl<'a> Transform<'a> {
    pub fn new(transform: &'a mut math::Transform) -> Self {
        Self { transform }
    }
}

impl egui::Widget for Transform<'_> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let mut add_3x_slider = |text: &str, v: &mut glm::DVec3| {
            let response_label = ui.label(text);
            let response_val_0 = ui.add(egui::Slider::new(&mut v[0], -5.0..=5.0));
            let response_val_1 = ui.add(egui::Slider::new(&mut v[1], -5.0..=5.0));
            let response_val_2 = ui.add(egui::Slider::new(&mut v[2], -5.0..=5.0));

            response_label.union(response_val_0.union(response_val_1.union(response_val_2)))
        };

        let response_location = add_3x_slider("Location", &mut self.transform.location);
        let response_rotation = add_3x_slider("Rotation", &mut self.transform.rotation);
        let response_scale = add_3x_slider("Scale", &mut self.transform.scale);

        response_location.union(response_rotation.union(response_scale))
    }
}
