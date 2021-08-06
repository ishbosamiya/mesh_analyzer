use std::{fmt::Display, marker::PhantomData, path::Path};

use itertools::Itertools;
use quick_renderer::{
    egui::{self, Color32},
    glm,
};
use serde::{Deserialize, Serialize};

use crate::{
    blender_mesh_io::{EmptyAdaptiveMesh, MeshExtensionError},
    draw_ui::DrawUI,
    math::{self, Transform},
    prelude::MeshExtension,
    ui_widgets,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Element {
    Node,
    Vert,
    Edge,
    Face,
}

impl Default for Element {
    fn default() -> Self {
        Self::Node
    }
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

type MeshType = EmptyAdaptiveMesh;
type ResultMesh = Result<MeshType, MeshExtensionError>;

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadedMesh {
    mesh: ResultMesh,
    location: String,
}

impl LoadedMesh {
    pub fn new(mesh: ResultMesh, location: String) -> Self {
        Self { mesh, location }
    }

    pub fn get_mesh(&self) -> &ResultMesh {
        &self.mesh
    }

    pub fn get_location(&self) -> &str {
        &self.location
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config<END, EVD, EED, EFD> {
    meshes_to_load: String,
    #[serde(skip)]
    meshes: Vec<LoadedMesh>,
    #[serde(skip)]
    mesh_index: usize,

    draw_wireframe: bool,
    draw_loose_edges: bool,
    #[serde(default = "default_draw_anisotropic_flippable_edges")]
    draw_anisotropic_flippable_edges: bool,

    #[serde(skip)]
    element: Element,
    #[serde(skip)]
    element_index: usize,

    mesh_transform: math::Transform,

    uv_plane_3d_transform: math::Transform,
    uv_map_color: glm::DVec4,

    node_color: glm::DVec4,
    vert_color: glm::DVec4,
    node_vert_connect_color: glm::DVec4,
    edge_color: glm::DVec4,
    loose_edge_color: glm::DVec4,
    #[serde(default = "default_anisotropic_flippable_edge_color")]
    anisotropic_flippable_edge_color: glm::DVec4,
    face_color: (glm::DVec4, glm::DVec4),
    normal_pull_factor: f64,

    mesh_node_extra_data_type: PhantomData<END>,
    mesh_vert_extra_data_type: PhantomData<EVD>,
    mesh_edge_extra_data_type: PhantomData<EED>,
    mesh_face_extra_data_type: PhantomData<EFD>,
}

fn default_draw_anisotropic_flippable_edges() -> bool {
    false
}

fn default_anisotropic_flippable_edge_color() -> glm::DVec4 {
    glm::vec4(0.5, 1.0, 0.4, 1.0)
}

impl<END, EVD, EED, EFD> Default for Config<END, EVD, EED, EFD> {
    fn default() -> Self {
        Self {
            meshes_to_load: "/tmp/adaptive_cloth/".to_string(),
            meshes: Vec::new(),
            mesh_index: 0,

            draw_wireframe: false,
            draw_loose_edges: false,
            draw_anisotropic_flippable_edges: default_draw_anisotropic_flippable_edges(),

            element: Element::Node,
            element_index: 0,

            mesh_transform: Default::default(),

            uv_plane_3d_transform: Default::default(),
            uv_map_color: glm::vec4(1.0, 1.0, 1.0, 1.0),

            node_color: glm::vec4(1.0, 0.78, 0.083, 1.0),
            vert_color: glm::vec4(0.48, 0.13, 1.0, 1.0),
            node_vert_connect_color: glm::vec4(0.17, 1.0, 0.4, 1.0),
            edge_color: glm::vec4(0.01, 0.52, 1.0, 1.0),
            loose_edge_color: glm::vec4(1.0, 0.2, 0.6, 1.0),
            anisotropic_flippable_edge_color: default_anisotropic_flippable_edge_color(),
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
    type ExtraData = ();

    fn draw_ui(&self, _extra_data: &Self::ExtraData, _ui: &mut egui::Ui) {}

    fn draw_ui_edit(&mut self, _extra_data: &Self::ExtraData, ui: &mut egui::Ui) {
        ui.text_edit_singleline(&mut self.meshes_to_load);

        ui.horizontal(|ui| {
            if ui.button("Load Mesh").clicked() {
                self.meshes = vec![LoadedMesh::new(
                    MeshType::read_file(&self.meshes_to_load),
                    self.meshes_to_load.to_string(),
                )];
            }
            if ui.button("Load Folder").clicked() {
                let path = Path::new(&self.meshes_to_load);

                if path.is_dir() {
                    self.meshes = path
                        .read_dir()
                        .unwrap()
                        .map(|location| {
                            let location = location.unwrap().path();
                            LoadedMesh::new(
                                MeshType::read_file(&location),
                                location.to_str().unwrap().to_string(),
                            )
                        })
                        .sorted_by(|lm1, lm2| Ord::cmp(lm1.get_location(), lm2.get_location()))
                        .collect();
                } else {
                    self.meshes = Vec::new();
                }
            }
            self.meshes = self
                .meshes
                .drain(..)
                .filter(|loaded_mesh| match loaded_mesh.get_mesh() {
                    Ok(_) => true,
                    Err(err) => !matches!(
                        err,
                        MeshExtensionError::FileExtensionUnknown
                            | MeshExtensionError::NoFileExtension
                    ),
                })
                .collect();
        });

        ui.add(
            egui::Slider::new(&mut self.mesh_index, 0..=(self.meshes.len().max(1) - 1))
                .clamp_to_range(true)
                .text("Mesh Index"),
        );

        let op_mesh = self.meshes.get(self.mesh_index);

        match op_mesh {
            Some(loaded_mesh) => match loaded_mesh.get_mesh() {
                Ok(mesh) => {
                    if ui.button("Save current mesh as OBJ").clicked() {
                        let location =
                            std::path::Path::new(loaded_mesh.get_location()).with_extension("obj");
                        mesh.write().write(location).unwrap();
                    }
                    ui.label(format!(
                        "Currently loaded mesh from file {}",
                        loaded_mesh.get_location()
                    ));
                    Some(mesh)
                }
                Err(err) => {
                    ui.label(format!(
                        "Error while loading mesh from {}: {}",
                        loaded_mesh.get_location(),
                        err
                    ));
                    None
                }
            },
            None => {
                ui.label(format!(
                    "Couldn't find mesh at index: {}, loaded {} meshes",
                    self.mesh_index,
                    self.meshes.len()
                ));
                None
            }
        };

        ui.checkbox(&mut self.draw_wireframe, "Draw Wireframe");
        ui.checkbox(&mut self.draw_loose_edges, "Draw Loose Edges");
        ui.checkbox(
            &mut self.draw_anisotropic_flippable_edges,
            "Draw Anisotropic Flippable Edges",
        );

        egui::ComboBox::from_label("Element Type")
            .selected_text(format!("{}", self.element))
            .show_ui(ui, |ui| {
                Element::all().for_each(|element| {
                    ui.selectable_value(&mut self.element, element, format!("{}", element));
                });
            });

        let mesh = match self.meshes.get(self.mesh_index) {
            Some(loaded_mesh) => loaded_mesh.get_mesh().as_ref().ok(),
            None => None,
        };

        let num_elements = mesh.map_or_else(
            || 1,
            |mesh| match self.element {
                Element::Node => mesh.get_nodes().len(),
                Element::Vert => mesh.get_verts().len(),
                Element::Edge => mesh.get_edges().len(),
                Element::Face => mesh.get_faces().len(),
            },
        );
        let num_elements = num_elements.max(1);
        ui.add(
            egui::Slider::new(&mut self.element_index, 0..=(num_elements - 1))
                .clamp_to_range(true)
                .text("Element Index"),
        );

        egui::Window::new("Element references").show(ui.ctx(), |ui| {
            if let Some(mesh) = mesh {
                let references_res =
                    mesh.get_references_in_element(self.get_element(), self.get_element_index());
                match references_res {
                    Ok(references) => {
                        references.draw_ui(&(), ui);
                    }
                    Err(err) => {
                        ui.label(format!("Got error: {}", err));
                    }
                }

                if self.draw_loose_edges {
                    ui.label(format!(
                        "Loose Edges: {:?}",
                        mesh.get_edges()
                            .iter()
                            .filter(|(_, edge)| edge.is_loose())
                            .map(|(_, edge)| { edge.get_self_index().0.into_raw_parts().0 })
                            .collect_vec()
                    ));
                }
            }
        });

        ui.separator();

        color_edit_button_dvec4(ui, "Fancy Node Color", &mut self.node_color);
        color_edit_button_dvec4(ui, "Fancy Vert Color", &mut self.vert_color);
        color_edit_button_dvec4(
            ui,
            "Fancy Node Vert Connect Color",
            &mut self.node_vert_connect_color,
        );
        color_edit_button_dvec4(ui, "Fancy Edge Color", &mut self.edge_color);
        color_edit_button_dvec4(ui, "Loose Fancy Edge Color", &mut self.loose_edge_color);
        color_edit_button_dvec4(
            ui,
            "Anisotropic Flippable Edge Fancy Edge Color",
            &mut self.anisotropic_flippable_edge_color,
        );
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
    #[allow(clippy::result_unit_err)]
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, ()> {
        let config_file = std::fs::read(path).map_err(|err| {
            println!("error reading file: {}", err);
        })?;
        let config: Self = serde_json::from_slice(&config_file).map_err(|err| {
            println!("error deserializing file: {}", err);
        })?;

        Ok(config)
    }

    pub fn get_mesh(&self) -> Result<&ResultMesh, MeshExtensionError> {
        self.meshes.get(self.mesh_index).map_or(
            Err(MeshExtensionError::NoElementAtIndex(self.mesh_index)),
            |loaded_mesh| Ok(loaded_mesh.get_mesh()),
        )
    }

    pub fn get_draw_wireframe(&self) -> bool {
        self.draw_wireframe
    }

    pub fn get_draw_loose_edges(&self) -> bool {
        self.draw_loose_edges
    }

    pub fn get_draw_anisotropic_flippable_edges(&self) -> bool {
        self.draw_anisotropic_flippable_edges
    }

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

    pub fn get_loose_edge_color(&self) -> glm::DVec4 {
        self.loose_edge_color
    }

    pub fn get_anisotropic_flippable_edge_color(&self) -> glm::DVec4 {
        self.anisotropic_flippable_edge_color
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
