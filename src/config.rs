use std::{fmt::Display, marker::PhantomData};

use quick_renderer::{egui, glm, mesh};

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
pub struct Config<END, EVD, EED, EFD> {
    element: Element,
    element_index: usize,

    uv_plane_3d_model_matrix: glm::DMat4,

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

            uv_plane_3d_model_matrix: glm::identity(),

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
    }
}

impl<END, EVD, EED, EFD> Config<END, EVD, EED, EFD> {
    pub fn get_element(&self) -> Element {
        self.element
    }

    pub fn get_element_index(&self) -> usize {
        self.element_index
    }

    pub fn get_uv_plane_3d_model_matrix(&self) -> &glm::DMat4 {
        &self.uv_plane_3d_model_matrix
    }
}
