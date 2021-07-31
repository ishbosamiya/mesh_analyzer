use std::{fmt::Display, marker::PhantomData};

use quick_renderer::{
    egui::{self, Color32},
    glm, mesh,
};

use crate::{
    draw_ui::DrawUI,
    math::{self, Transform},
    ui_widgets,
};

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

#[derive(Debug, Clone)]
pub struct Config<END, EVD, EED, EFD> {
    element: Element,
    element_index: usize,

    mesh_transform: math::Transform,

    uv_plane_3d_transform: math::Transform,
    uv_map_color: glm::DVec4,

    node_color: glm::DVec4,
    vert_color: glm::DVec4,
    node_vert_connect_color: glm::DVec4,
    edge_color: glm::DVec4,
    face_color: (glm::DVec4, glm::DVec4),
    normal_pull_factor: f64,

    mesh_node_extra_data_type: PhantomData<END>,
    mesh_vert_extra_data_type: PhantomData<EVD>,
    mesh_edge_extra_data_type: PhantomData<EED>,
    mesh_face_extra_data_type: PhantomData<EFD>,
}

impl<END, EVD, EED, EFD> Default for Config<END, EVD, EED, EFD> {
    fn default() -> Self {
        Self {
            element: Element::Node,
            element_index: 0,

            mesh_transform: Default::default(),

            uv_plane_3d_transform: Default::default(),
            uv_map_color: glm::vec4(1.0, 1.0, 1.0, 1.0),

            node_color: glm::vec4(1.0, 0.78, 0.083, 1.0),
            vert_color: glm::vec4(0.48, 0.13, 1.0, 1.0),
            node_vert_connect_color: glm::vec4(0.17, 1.0, 0.4, 1.0),
            edge_color: glm::vec4(0.01, 0.52, 1.0, 1.0),
            face_color: (
                glm::vec4(1.0, 0.17, 0.01, 0.4),
                glm::vec4(0.07, 1.0, 0.4, 0.4),
            ),
            normal_pull_factor: 0.2,

            mesh_node_extra_data_type: PhantomData,
            mesh_vert_extra_data_type: PhantomData,
            mesh_edge_extra_data_type: PhantomData,
            mesh_face_extra_data_type: PhantomData,
        }
    }
}

impl<END, EVD, EED, EFD> DrawUI for Config<END, EVD, EED, EFD> {
    type ExtraData = mesh::Mesh<END, EVD, EED, EFD>;

    fn draw_ui(&self, _extra_data: &Self::ExtraData, _ui: &mut egui::Ui) {}

    fn draw_ui_edit(&mut self, extra_data: &Self::ExtraData, ui: &mut egui::Ui) {
        let mesh = extra_data;
        egui::ComboBox::from_label("Element Type")
            .selected_text(format!("{}", self.element))
            .show_ui(ui, |ui| {
                Element::all().for_each(|element| {
                    ui.selectable_value(&mut self.element, element, format!("{}", element));
                });
            });

        let num_elements = match self.element {
            Element::Node => mesh.get_nodes().len(),
            Element::Vert => mesh.get_verts().len(),
            Element::Edge => mesh.get_edges().len(),
            Element::Face => mesh.get_faces().len(),
        };
        ui.add(
            egui::Slider::new(&mut self.element_index, 0..=(num_elements - 1))
                .clamp_to_range(true)
                .text("Element Index"),
        );

        ui.separator();

        color_edit_button_dvec4(ui, "Fancy Node Color", &mut self.node_color);
        color_edit_button_dvec4(ui, "Fancy Vert Color", &mut self.vert_color);
        color_edit_button_dvec4(
            ui,
            "Fancy Node Vert Connect Color",
            &mut self.node_vert_connect_color,
        );
        color_edit_button_dvec4(ui, "Fancy Edge Color", &mut self.edge_color);
        color_edit_button_dvec4_range(ui, "Fancy Face Color Range", &mut self.face_color);
        ui.add(
            egui::Slider::new(&mut self.normal_pull_factor, 0.0..=3.0)
                .text("Bendiness of the fancy edges and faces"),
        );

        ui.separator();

        ui.label("UV Plane Transform");
        ui.add(ui_widgets::Transform::new(&mut self.uv_plane_3d_transform));
        color_edit_button_dvec4(ui, "UV Map Color", &mut self.uv_map_color);

        ui.separator();

        ui.label("Mesh Transform");
        ui.add(ui_widgets::Transform::new(&mut self.mesh_transform));

        ui.separator();
    }
}

impl<END, EVD, EED, EFD> Config<END, EVD, EED, EFD> {
    pub fn get_element(&self) -> Element {
        self.element
    }

    pub fn get_element_index(&self) -> usize {
        self.element_index
    }

    pub fn get_uv_plane_3d_transform(&self) -> &Transform {
        &self.uv_plane_3d_transform
    }

    pub fn get_uv_map_color(&self) -> glm::DVec4 {
        self.uv_map_color
    }

    pub fn get_node_color(&self) -> glm::DVec4 {
        self.node_color
    }

    pub fn get_vert_color(&self) -> glm::DVec4 {
        self.vert_color
    }

    pub fn get_node_vert_connect_color(&self) -> glm::DVec4 {
        self.node_vert_connect_color
    }

    pub fn get_edge_color(&self) -> glm::DVec4 {
        self.edge_color
    }

    pub fn get_face_color(&self) -> (glm::DVec4, glm::DVec4) {
        self.face_color
    }

    pub fn get_normal_pull_factor(&self) -> f64 {
        self.normal_pull_factor
    }

    pub fn get_mesh_transform(&self) -> &Transform {
        &self.mesh_transform
    }
}

fn color_edit_dvec4(ui: &mut egui::Ui, color: &mut glm::DVec4) {
    let mut color_egui = Color32::from_rgba_premultiplied(
        (color[0] * 255.0) as _,
        (color[1] * 255.0) as _,
        (color[2] * 255.0) as _,
        (color[3] * 255.0) as _,
    );
    egui::color_picker::color_edit_button_srgba(
        ui,
        &mut color_egui,
        egui::color_picker::Alpha::BlendOrAdditive,
    );
    *color = glm::vec4(
        color_egui.r() as f64 / 255.0,
        color_egui.g() as f64 / 255.0,
        color_egui.b() as f64 / 255.0,
        color_egui.a() as f64 / 255.0,
    );
}

fn color_edit_button_dvec4(ui: &mut egui::Ui, text: &str, color: &mut glm::DVec4) {
    ui.horizontal(|ui| {
        ui.label(text);
        color_edit_dvec4(ui, color);
    });
}

fn color_edit_button_dvec4_range(
    ui: &mut egui::Ui,
    text: &str,
    color: &mut (glm::DVec4, glm::DVec4),
) {
    ui.horizontal(|ui| {
        ui.label(text);
        color_edit_dvec4(ui, &mut color.0);
        color_edit_dvec4(ui, &mut color.1);
    });
}
