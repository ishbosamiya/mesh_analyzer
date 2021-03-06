use std::{fmt::Display, marker::PhantomData, path::Path};

use itertools::Itertools;
use quick_renderer::{
    egui::{self, Color32},
    glm,
    mesh::{apply_model_matrix_vec2, apply_model_matrix_vec3, MeshUseShader},
};
use serde::{Deserialize, Serialize};

use crate::{
    blender_mesh_io::{AdaptiveMeshExtension, MeshExtensionError, MeshToBlenderMeshIndexMap},
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClothVertexElements {
    Flags,
    V,
    Xconst,
    X,
    Xold,
    Tx,
    Txold,
    Tv,
    Mass,
    Goal,
    Impulse,
    Xrest,
    Dcvel,
    ImpulseCount,
    AvgSpringLen,
    StructStiff,
    BendStiff,
    ShearStiff,
    SpringCount,
    ShrinkFactor,
    InternalStiff,
    PressureFactor,
}

impl ClothVertexElements {
    pub fn all() -> impl Iterator<Item = Self> {
        [
            Self::Flags,
            Self::V,
            Self::Xconst,
            Self::X,
            Self::Xold,
            Self::Tx,
            Self::Txold,
            Self::Tv,
            Self::Mass,
            Self::Goal,
            Self::Impulse,
            Self::Xrest,
            Self::Dcvel,
            Self::ImpulseCount,
            Self::AvgSpringLen,
            Self::StructStiff,
            Self::BendStiff,
            Self::ShearStiff,
            Self::SpringCount,
            Self::ShrinkFactor,
            Self::InternalStiff,
            Self::PressureFactor,
        ]
        .iter()
        .copied()
    }
}

impl Display for ClothVertexElements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Flags => write!(f, "Flags"),
            Self::V => write!(f, "V"),
            Self::Xconst => write!(f, "Xconst"),
            Self::X => write!(f, "X"),
            Self::Xold => write!(f, "Xold"),
            Self::Tx => write!(f, "Tx"),
            Self::Txold => write!(f, "Txold"),
            Self::Tv => write!(f, "Tv"),
            Self::Mass => write!(f, "Mass"),
            Self::Goal => write!(f, "Goal"),
            Self::Impulse => write!(f, "Impulse"),
            Self::Xrest => write!(f, "Xrest"),
            Self::Dcvel => write!(f, "Dcvel"),
            Self::ImpulseCount => write!(f, "Impulse Count"),
            Self::AvgSpringLen => write!(f, "Avg Spring Len"),
            Self::StructStiff => write!(f, "Struct Stiff"),
            Self::BendStiff => write!(f, "Bend Stiff"),
            Self::ShearStiff => write!(f, "Shear Stiff"),
            Self::SpringCount => write!(f, "Spring Count"),
            Self::ShrinkFactor => write!(f, "Shrink Factor"),
            Self::InternalStiff => write!(f, "Internal Stiff"),
            Self::PressureFactor => write!(f, "Pressure Factor"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AspectRatioMetric {
    MeasureWithInterpolationError,
    RatioBetweenMinMaxDimension,
    DifferentMetric,
}

impl AspectRatioMetric {
    pub fn all() -> impl Iterator<Item = Self> {
        [
            Self::MeasureWithInterpolationError,
            Self::RatioBetweenMinMaxDimension,
            Self::DifferentMetric,
        ]
        .iter()
        .copied()
    }
}

impl Display for AspectRatioMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MeasureWithInterpolationError => {
                write!(f, "Measure associated with interpolation error")
            }
            AspectRatioMetric::RatioBetweenMinMaxDimension => {
                write!(f, "Ratio between the min and max dimensions")
            }
            AspectRatioMetric::DifferentMetric => write!(f, "Different metric"),
        }
    }
}

#[cfg(not(feature = "use_cloth_adaptive_mesh"))]
use crate::blender_mesh_io::EmptyAdaptiveMesh;
#[cfg(not(feature = "use_cloth_adaptive_mesh"))]
type MeshType = EmptyAdaptiveMesh;

#[cfg(feature = "use_cloth_adaptive_mesh")]
use crate::blender_mesh_io::ClothAdaptiveMesh;
#[cfg(feature = "use_cloth_adaptive_mesh")]
type MeshType = ClothAdaptiveMesh;

type ResultMesh = Result<MeshType, MeshExtensionError>;

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadedMesh {
    mesh: ResultMesh,
    mesh_pos_index_map: Option<MeshToBlenderMeshIndexMap>,
    location: String,
}

impl LoadedMesh {
    pub fn new(
        mesh: ResultMesh,
        mesh_pos_index_map: Option<MeshToBlenderMeshIndexMap>,
        location: String,
    ) -> Self {
        Self {
            mesh,
            mesh_pos_index_map,
            location,
        }
    }

    pub fn get_mesh(&self) -> &ResultMesh {
        &self.mesh
    }

    pub fn get_mesh_pos_index_map(&self) -> &Option<MeshToBlenderMeshIndexMap> {
        &self.mesh_pos_index_map
    }

    pub fn get_location(&self) -> &str {
        &self.location
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EigenDecompType {
    GLM,
    Custom,
}

impl Default for EigenDecompType {
    fn default() -> Self {
        EigenDecompType::GLM
    }
}

impl Display for EigenDecompType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EigenDecompType::GLM => write!(f, "GLM"),
            EigenDecompType::Custom => write!(f, "Custom"),
        }
    }
}

impl EigenDecompType {
    pub fn all() -> impl Iterator<Item = Self> {
        [EigenDecompType::GLM, EigenDecompType::Custom]
            .iter()
            .copied()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRemeshParams {
    edge_length_min: f64,
    edge_length_max: f64,
    aspect_ratio_min: f64,
    change_in_vertex_normal_max: f64,
    #[serde(default = "default_change_in_vertex_velocity_max")]
    change_in_vertex_velocity_max: f64,
    #[serde(default = "default_material_compression_max")]
    material_compression_max: f64,
    #[serde(default)]
    eigen_decomp_type: EigenDecompType,
}

fn default_change_in_vertex_velocity_max() -> f64 {
    0.3
}

fn default_material_compression_max() -> f64 {
    0.3
}

impl Default for AdaptiveRemeshParams {
    fn default() -> Self {
        Self::new(
            0.05,
            0.5,
            0.2,
            0.3,
            default_change_in_vertex_velocity_max(),
            default_material_compression_max(),
            EigenDecompType::default(),
        )
    }
}

impl AdaptiveRemeshParams {
    pub fn new(
        edge_length_min: f64,
        edge_length_max: f64,
        aspect_ratio_min: f64,
        change_in_vertex_normal_max: f64,
        change_in_vertex_velocity_max: f64,
        material_compression_max: f64,
        eigen_decomp_type: EigenDecompType,
    ) -> Self {
        Self {
            edge_length_min,
            edge_length_max,
            aspect_ratio_min,
            change_in_vertex_normal_max,
            change_in_vertex_velocity_max,
            material_compression_max,
            eigen_decomp_type,
        }
    }

    pub fn get_edge_length_min(&self) -> f64 {
        self.edge_length_min
    }

    pub fn get_edge_length_max(&self) -> f64 {
        self.edge_length_max
    }

    pub fn get_aspect_ratio_min(&self) -> f64 {
        self.aspect_ratio_min
    }

    pub fn get_change_in_vertex_normal_max(&self) -> f64 {
        self.change_in_vertex_normal_max
    }

    pub fn get_change_in_vertex_velocity_max(&self) -> f64 {
        self.change_in_vertex_velocity_max
    }

    pub fn get_material_compression_max(&self) -> f64 {
        self.material_compression_max
    }

    pub fn get_eigen_decomp_type(&self) -> EigenDecompType {
        self.eigen_decomp_type
    }

    pub fn get_edge_length_min_mut(&mut self) -> &mut f64 {
        &mut self.edge_length_min
    }

    pub fn get_edge_length_max_mut(&mut self) -> &mut f64 {
        &mut self.edge_length_max
    }

    pub fn get_aspect_ratio_min_mut(&mut self) -> &mut f64 {
        &mut self.aspect_ratio_min
    }

    pub fn get_change_in_vertex_normal_max_mut(&mut self) -> &mut f64 {
        &mut self.change_in_vertex_normal_max
    }

    pub fn get_change_in_vertex_velocity_max_mut(&mut self) -> &mut f64 {
        &mut self.change_in_vertex_velocity_max
    }

    pub fn get_material_compression_max_mut(&mut self) -> &mut f64 {
        &mut self.material_compression_max
    }

    pub fn get_eigen_decomp_type_mut(&mut self) -> &mut EigenDecompType {
        &mut self.eigen_decomp_type
    }
}

impl DrawUI for AdaptiveRemeshParams {
    type ExtraData = ();

    fn draw_ui(&self, _extra_data: &Self::ExtraData, ui: &mut egui::Ui) {
        ui.label(format!(
            "Minimum Edge Length: {}",
            self.get_edge_length_min()
        ));
        ui.label(format!(
            "Maximum Edge Length: {}",
            self.get_edge_length_max()
        ));
        ui.label(format!(
            "Minimum Aspect Ratio: {}",
            self.get_aspect_ratio_min()
        ));
        ui.label(format!(
            "Maximum Change in Vertex Normal: {}",
            self.get_change_in_vertex_normal_max()
        ));
        ui.label(format!(
            "Maximum Change in Vertex Velocity: {}",
            self.get_change_in_vertex_velocity_max()
        ));
        ui.label(format!(
            "Maximum Material Compression: {}",
            self.get_material_compression_max()
        ));
        ui.label(format!(
            "Eigen Decomp Type: {}",
            self.get_eigen_decomp_type()
        ));
    }

    fn draw_ui_edit(&mut self, _extra_data: &Self::ExtraData, ui: &mut egui::Ui) {
        ui.add(
            egui::Slider::new(self.get_edge_length_min_mut(), 0.0..=1.0)
                .text("Minimum Edge Length"),
        );
        ui.add(
            egui::Slider::new(self.get_edge_length_max_mut(), 0.0..=1.0)
                .text("Maximum Edge Length"),
        );
        ui.add(
            egui::Slider::new(self.get_aspect_ratio_min_mut(), 0.0..=1.0)
                .text("Minimum Aspect Ratio"),
        );
        ui.add(
            egui::Slider::new(self.get_change_in_vertex_normal_max_mut(), 0.0..=1.0)
                .text("Maximum Change in Vertex Normal"),
        );
        ui.add(
            egui::Slider::new(self.get_change_in_vertex_velocity_max_mut(), 0.0..=1.0)
                .text("Maximum Change in Vertex Velocity"),
        );
        ui.add(
            egui::Slider::new(self.get_material_compression_max_mut(), 0.0..=1.0)
                .text("Maximum Material Compression"),
        );
        egui::ComboBox::from_label("Eigen Decomp Type")
            .selected_text(format!("{}", self.eigen_decomp_type))
            .show_ui(ui, |ui| {
                EigenDecompType::all().for_each(|element| {
                    ui.selectable_value(
                        &mut self.eigen_decomp_type,
                        element,
                        format!("{}", element),
                    );
                });
            });
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config<END, EVD, EED, EFD> {
    meshes_to_load: String,
    #[serde(skip)]
    meshes: Vec<LoadedMesh>,
    #[serde(skip)]
    mesh_index: usize,

    #[serde(default = "default_draw_mesh_with_shader")]
    draw_mesh_with_shader: MeshUseShader,

    #[serde(default = "default_draw_infinite_grid")]
    draw_infinite_grid: bool,
    #[serde(default = "default_draw_face_normals")]
    draw_face_normals: bool,
    #[serde(default = "default_draw_node_normals")]
    draw_node_normals: bool,
    #[serde(default = "default_draw_computed_node_normals")]
    draw_computed_node_normals: bool,
    draw_wireframe: bool,
    #[serde(default = "default_draw_loose_nodes")]
    draw_loose_nodes: bool,
    #[serde(default = "default_draw_loose_verts")]
    draw_loose_verts: bool,
    draw_loose_edges: bool,
    #[serde(default = "default_show_loose_faces")]
    show_loose_faces: bool,
    #[serde(default = "default_draw_anisotropic_flippable_edges")]
    draw_anisotropic_flippable_edges: bool,
    #[serde(default = "default_show_aspect_ratios_of_faces")]
    show_aspect_ratios_of_faces: bool,
    #[serde(default = "default_draw_faces_violating_aspect_ratio")]
    draw_faces_violating_aspect_ratio: bool,
    #[serde(default = "default_draw_cloth_vertex_data")]
    draw_cloth_vertex_data: bool,
    #[serde(default = "default_draw_edge_between_sewing_edges")]
    draw_edge_between_sewing_edges: bool,
    #[serde(default = "default_show_node_data")]
    show_node_data: bool,
    #[serde(default = "default_show_vert_data")]
    show_vert_data: bool,
    #[serde(default = "default_show_edge_data_flags")]
    show_edge_data_flags: bool,
    #[serde(default = "default_show_edge_data")]
    show_edge_data: bool,
    #[serde(default = "default_show_computed_face_sizing")]
    show_computed_face_sizing: bool,
    #[serde(default = "default_modify_adaptive_remesh_params")]
    modify_adaptive_remesh_params: bool,

    #[serde(default)]
    element: Element,
    #[serde(skip)]
    element_index: usize,
    #[serde(skip)]
    blender_element_index: String,

    mesh_transform: math::Transform,

    uv_plane_3d_transform: math::Transform,
    uv_map_color: glm::DVec4,

    node_color: glm::DVec4,
    vert_color: glm::DVec4,
    node_vert_connect_color: glm::DVec4,
    edge_color: glm::DVec4,
    #[serde(default = "default_loose_node_color")]
    loose_node_color: glm::DVec4,
    #[serde(default = "default_loose_vert_color")]
    loose_vert_color: glm::DVec4,
    loose_edge_color: glm::DVec4,
    #[serde(default = "default_anisotropic_flippable_edge_color")]
    anisotropic_flippable_edge_color: glm::DVec4,
    face_color: (glm::DVec4, glm::DVec4),
    #[serde(default = "default_face_front_color")]
    face_front_color: glm::DVec4,
    #[serde(default = "default_face_back_color")]
    face_back_color: glm::DVec4,
    #[serde(default = "default_face_normal_color")]
    face_normal_color: glm::DVec4,
    #[serde(default = "default_node_normal_color")]
    node_normal_color: glm::DVec4,
    #[serde(default = "default_computed_node_normal_color")]
    computed_node_normal_color: glm::DVec4,
    #[serde(default = "default_face_violating_aspect_ratio_color")]
    face_violating_aspect_ratio_color: glm::DVec4,
    #[serde(default = "default_cloth_vertex_data_color")]
    cloth_vertex_data_color: glm::DVec4,
    #[serde(default = "default_edge_between_sewing_edges_color")]
    edge_between_sewing_edges_color: glm::DVec4,
    #[serde(default = "default_edge_data_flags_color")]
    edge_data_flags_color: glm::DVec4,

    #[serde(default = "default_face_normal_size")]
    face_normal_size: f64,
    #[serde(default = "default_node_normal_size")]
    node_normal_size: f64,
    #[serde(default = "default_computed_node_normal_size")]
    computed_node_normal_size: f64,
    normal_pull_factor: f64,
    #[serde(default = "default_aspect_ratio_min")]
    aspect_ratio_min: f64,
    #[serde(default = "default_adaptive_remesh_params")]
    adaptive_remesh_params: AdaptiveRemeshParams,

    #[serde(default = "default_aspect_ratio_metric")]
    aspect_ratio_metric: AspectRatioMetric,
    #[serde(default = "default_cloth_vertex_elements")]
    cloth_vertex_elements: ClothVertexElements,

    #[serde(default = "default_fps_limit")]
    fps_limit: f64,

    mesh_node_extra_data_type: PhantomData<END>,
    mesh_vert_extra_data_type: PhantomData<EVD>,
    mesh_edge_extra_data_type: PhantomData<EED>,
    mesh_face_extra_data_type: PhantomData<EFD>,
}

fn default_draw_infinite_grid() -> bool {
    true
}

fn default_draw_face_normals() -> bool {
    false
}

fn default_draw_node_normals() -> bool {
    false
}

fn default_draw_computed_node_normals() -> bool {
    false
}

fn default_draw_loose_nodes() -> bool {
    false
}

fn default_draw_loose_verts() -> bool {
    false
}

fn default_show_loose_faces() -> bool {
    false
}

fn default_loose_node_color() -> glm::DVec4 {
    glm::vec4(1.0, 0.2, 0.6, 1.0)
}

fn default_loose_vert_color() -> glm::DVec4 {
    glm::vec4(1.0, 0.2, 0.6, 1.0)
}

fn default_face_front_color() -> glm::DVec4 {
    glm::vec4(0.21, 0.42, 1.0, 1.0)
}

fn default_face_back_color() -> glm::DVec4 {
    glm::vec4(0.94, 0.22, 0.22, 1.0)
}

fn default_face_normal_color() -> glm::DVec4 {
    glm::vec4(1.0, 0.22, 0.22, 1.0)
}

fn default_node_normal_color() -> glm::DVec4 {
    glm::vec4(1.0, 0.22, 0.22, 1.0)
}

fn default_computed_node_normal_color() -> glm::DVec4 {
    glm::vec4(1.0, 0.22, 0.22, 1.0)
}

fn default_face_violating_aspect_ratio_color() -> glm::DVec4 {
    glm::vec4(1.0, 0.22, 0.58, 1.0)
}

fn default_cloth_vertex_data_color() -> glm::DVec4 {
    glm::vec4(0.1, 0.6, 0.8, 1.0)
}

fn default_edge_between_sewing_edges_color() -> glm::DVec4 {
    glm::vec4(0.71, 0.92, 0.27, 1.0)
}

fn default_edge_data_flags_color() -> glm::DVec4 {
    glm::vec4(0.31, 0.57, 0.92, 1.0)
}

fn default_face_normal_size() -> f64 {
    1.0
}

fn default_node_normal_size() -> f64 {
    1.0
}

fn default_computed_node_normal_size() -> f64 {
    1.0
}

fn default_draw_mesh_with_shader() -> MeshUseShader {
    MeshUseShader::DirectionalLight
}

fn default_draw_anisotropic_flippable_edges() -> bool {
    false
}

fn default_anisotropic_flippable_edge_color() -> glm::DVec4 {
    glm::vec4(0.5, 1.0, 0.4, 1.0)
}

fn default_show_aspect_ratios_of_faces() -> bool {
    false
}

fn default_draw_faces_violating_aspect_ratio() -> bool {
    false
}

fn default_draw_cloth_vertex_data() -> bool {
    false
}

fn default_draw_edge_between_sewing_edges() -> bool {
    false
}

fn default_show_node_data() -> bool {
    false
}

fn default_show_vert_data() -> bool {
    false
}

fn default_show_edge_data_flags() -> bool {
    false
}

fn default_show_edge_data() -> bool {
    false
}

fn default_show_computed_face_sizing() -> bool {
    false
}

fn default_modify_adaptive_remesh_params() -> bool {
    false
}

fn default_aspect_ratio_min() -> f64 {
    0.1
}

fn default_adaptive_remesh_params() -> AdaptiveRemeshParams {
    Default::default()
}

fn default_aspect_ratio_metric() -> AspectRatioMetric {
    AspectRatioMetric::MeasureWithInterpolationError
}

fn default_cloth_vertex_elements() -> ClothVertexElements {
    ClothVertexElements::X
}

fn default_fps_limit() -> f64 {
    60.0
}

impl<END, EVD, EED, EFD> Default for Config<END, EVD, EED, EFD> {
    fn default() -> Self {
        Self {
            meshes_to_load: "/tmp/adaptive_cloth/".to_string(),
            meshes: Vec::new(),
            mesh_index: 0,

            draw_mesh_with_shader: default_draw_mesh_with_shader(),

            draw_infinite_grid: default_draw_infinite_grid(),
            draw_face_normals: default_draw_face_normals(),
            draw_node_normals: default_draw_node_normals(),
            draw_computed_node_normals: default_draw_computed_node_normals(),
            draw_wireframe: false,
            draw_loose_nodes: default_draw_loose_nodes(),
            draw_loose_verts: default_draw_loose_verts(),
            draw_loose_edges: false,
            show_loose_faces: default_show_loose_faces(),
            draw_anisotropic_flippable_edges: default_draw_anisotropic_flippable_edges(),
            show_aspect_ratios_of_faces: default_show_aspect_ratios_of_faces(),
            draw_faces_violating_aspect_ratio: default_draw_faces_violating_aspect_ratio(),
            draw_cloth_vertex_data: default_draw_cloth_vertex_data(),
            draw_edge_between_sewing_edges: default_draw_edge_between_sewing_edges(),
            show_node_data: default_show_node_data(),
            show_vert_data: default_show_vert_data(),
            show_edge_data_flags: default_show_edge_data_flags(),
            show_edge_data: default_show_edge_data(),
            show_computed_face_sizing: default_show_computed_face_sizing(),
            modify_adaptive_remesh_params: default_modify_adaptive_remesh_params(),

            element: Element::Node,
            element_index: 0,
            blender_element_index: String::new(),

            mesh_transform: Default::default(),

            uv_plane_3d_transform: Default::default(),
            uv_map_color: glm::vec4(1.0, 1.0, 1.0, 1.0),

            node_color: glm::vec4(1.0, 0.78, 0.083, 1.0),
            vert_color: glm::vec4(0.48, 0.13, 1.0, 1.0),
            node_vert_connect_color: glm::vec4(0.17, 1.0, 0.4, 1.0),
            edge_color: glm::vec4(0.01, 0.52, 1.0, 1.0),
            loose_node_color: default_loose_node_color(),
            loose_vert_color: default_loose_vert_color(),
            loose_edge_color: glm::vec4(1.0, 0.2, 0.6, 1.0),
            anisotropic_flippable_edge_color: default_anisotropic_flippable_edge_color(),
            face_color: (
                glm::vec4(1.0, 0.17, 0.01, 0.4),
                glm::vec4(0.07, 1.0, 0.4, 0.4),
            ),
            face_front_color: default_face_front_color(),
            face_back_color: default_face_back_color(),
            face_normal_color: default_face_normal_color(),
            node_normal_color: default_node_normal_color(),
            computed_node_normal_color: default_computed_node_normal_color(),
            face_violating_aspect_ratio_color: default_face_violating_aspect_ratio_color(),
            cloth_vertex_data_color: default_cloth_vertex_data_color(),
            edge_between_sewing_edges_color: default_edge_between_sewing_edges_color(),
            edge_data_flags_color: default_edge_data_flags_color(),

            normal_pull_factor: 0.2,
            face_normal_size: default_face_normal_size(),
            node_normal_size: default_node_normal_size(),
            computed_node_normal_size: default_computed_node_normal_size(),
            aspect_ratio_min: default_aspect_ratio_min(),
            adaptive_remesh_params: default_adaptive_remesh_params(),

            aspect_ratio_metric: default_aspect_ratio_metric(),
            cloth_vertex_elements: default_cloth_vertex_elements(),

            fps_limit: default_fps_limit(),

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
        ui.add(
            egui::Slider::new(&mut self.fps_limit, 60.0..=1000.0)
                .clamp_to_range(true)
                .text("FPS Limit"),
        );

        ui.text_edit_singleline(&mut self.meshes_to_load);

        ui.horizontal(|ui| {
            if ui.button("Load Mesh").clicked() {
                let mesh_maybe = MeshType::read_file(&self.meshes_to_load);
                let mesh_pos_index_map = mesh_maybe
                    .as_ref()
                    .map(|(_, mesh_pos_index_map)| mesh_pos_index_map.clone())
                    .ok();
                self.meshes = vec![LoadedMesh::new(
                    mesh_maybe.map(|(mesh, _)| mesh),
                    mesh_pos_index_map,
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
                            let mesh_maybe = MeshType::read_file(&location);
                            let mesh_pos_index_map = mesh_maybe
                                .as_ref()
                                .map(|(_, mesh_pos_index_map)| mesh_pos_index_map.clone())
                                .ok();
                            LoadedMesh::new(
                                mesh_maybe.map(|(mesh, _)| mesh),
                                mesh_pos_index_map,
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

        egui::ComboBox::from_label("Mesh Shader Type")
            .selected_text(format!("{}", self.draw_mesh_with_shader))
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.draw_mesh_with_shader,
                    MeshUseShader::DirectionalLight,
                    format!("{}", MeshUseShader::DirectionalLight),
                );
                ui.selectable_value(
                    &mut self.draw_mesh_with_shader,
                    MeshUseShader::FaceOrientation,
                    format!("{}", MeshUseShader::FaceOrientation),
                );
            });

        ui.checkbox(&mut self.draw_infinite_grid, "Draw Floor Grid");
        ui.checkbox(&mut self.draw_face_normals, "Draw Face Normals");
        ui.checkbox(&mut self.draw_node_normals, "Draw Node Normals");
        ui.checkbox(
            &mut self.draw_computed_node_normals,
            "Draw Computed Node Normals",
        );
        ui.checkbox(&mut self.draw_wireframe, "Draw Wireframe");
        ui.checkbox(&mut self.draw_loose_nodes, "Draw Loose Nodes");
        ui.checkbox(&mut self.draw_loose_verts, "Draw Loose Verts");
        ui.checkbox(&mut self.draw_loose_edges, "Draw Loose Edges");
        ui.checkbox(&mut self.show_loose_faces, "Show Loose Faces");
        ui.checkbox(
            &mut self.draw_anisotropic_flippable_edges,
            "Draw Anisotropic Flippable Edges",
        );
        ui.checkbox(
            &mut self.show_aspect_ratios_of_faces,
            "Show Aspect Ratios of Faces",
        );
        ui.checkbox(
            &mut self.draw_faces_violating_aspect_ratio,
            "Draw Faces Violating Aspect Ratio Minimum",
        );
        ui.checkbox(&mut self.draw_cloth_vertex_data, "Draw Cloth Vertex Data");
        ui.checkbox(
            &mut self.draw_edge_between_sewing_edges,
            "Draw Edge Between Sewing Edges",
        );
        ui.checkbox(&mut self.show_node_data, "Show Node Data");
        ui.checkbox(&mut self.show_vert_data, "Show Vert Data");
        ui.checkbox(&mut self.show_edge_data_flags, "Show Edge Data Flags");
        ui.checkbox(&mut self.show_edge_data, "Show Edge Data");
        ui.checkbox(
            &mut self.show_computed_face_sizing,
            "Show Computed Face Sizing",
        );
        ui.checkbox(
            &mut self.modify_adaptive_remesh_params,
            "Modify Adaptive Remesh Params",
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

        ui.horizontal(|ui| {
            ui.text_edit_singleline(&mut self.blender_element_index);
            if ui.button("Load Blender Element Position").clicked() {
                if let Ok(pos_index) = self.blender_element_index.parse::<usize>() {
                    if let Some(loaded_mesh) = self.meshes.get(self.mesh_index) {
                        if let Some(map) = loaded_mesh.get_mesh_pos_index_map() {
                            self.element_index = match self.element {
                                Element::Node => map
                                    .get_node_pos_index_map()
                                    .iter()
                                    .find(|(index, _)| index.get_index() == pos_index)
                                    .map(|(_, pos)| *pos)
                                    .unwrap_or(self.element_index),
                                Element::Vert => map
                                    .get_vert_pos_index_map()
                                    .iter()
                                    .find(|(index, _)| index.get_index() == pos_index)
                                    .map(|(_, pos)| *pos)
                                    .unwrap_or(self.element_index),
                                Element::Edge => map
                                    .get_edge_pos_index_map()
                                    .iter()
                                    .find(|(index, _)| index.get_index() == pos_index)
                                    .map(|(_, pos)| *pos)
                                    .unwrap_or(self.element_index),
                                Element::Face => map
                                    .get_face_pos_index_map()
                                    .iter()
                                    .find(|(index, _)| index.get_index() == pos_index)
                                    .map(|(_, pos)| *pos)
                                    .unwrap_or(self.element_index),
                            }
                        }
                    }
                }
            }
        });

        ui.separator();

        let mesh = match self.meshes.get(self.mesh_index) {
            Some(loaded_mesh) => loaded_mesh.get_mesh().as_ref().ok(),
            None => None,
        };

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

                if self.draw_loose_nodes {
                    ui.label(format!(
                        "Loose Nodes: {:?}",
                        mesh.get_nodes()
                            .iter()
                            .filter(|(_, node)| node.is_loose())
                            .map(|(_, node)| { node.get_self_index().0.into_raw_parts().0 })
                            .collect_vec()
                    ));
                }

                if self.draw_loose_verts {
                    ui.label(format!(
                        "Loose Verts: {:?}",
                        mesh.get_verts()
                            .iter()
                            .filter(|(_, vert)| vert.is_loose())
                            .map(|(_, vert)| { vert.get_self_index().0.into_raw_parts().0 })
                            .collect_vec()
                    ));
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

                if self.show_loose_faces {
                    ui.label(format!(
                        "Loose Faces: {:?}",
                        mesh.get_faces()
                            .iter()
                            .filter(|(_, face)| face.get_verts().is_empty())
                            .map(|(_, face)| { face.get_self_index().0.into_raw_parts().0 })
                            .collect_vec()
                    ));
                }
            }
        });

        egui::Window::new("Aspect Ratios of Triangles")
            .open(&mut self.show_aspect_ratios_of_faces)
            .show(ui.ctx(), |ui| {
                ui.scope(|ui| {
                    ui.label("Metric 1: The measure associated with interpolation error");
                    ui.label(
                    "Metric 2: Aspect ratio or ratio between min and max dimensions of triangle",
                );
                    ui.label("Metric 3: Different metric");
                });
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if let Some(mesh) = mesh {
                        for (_, face) in mesh.get_faces() {
                            let aspect_ratio = mesh.compute_aspect_ratio_uv(face);

                            ui.label(format!(
                                "{:.2}\t{:.2}\t{:.2}",
                                aspect_ratio.0, aspect_ratio.1, aspect_ratio.2
                            ));
                        }
                    }
                });
            });

        egui::Window::new("Node Data")
            .open(&mut self.show_node_data)
            .show(ui.ctx(), |ui| {
                if let Some(mesh) = mesh {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        mesh.draw_ui_node_data(ui);
                    });
                }
            });

        egui::Window::new("Vert Data")
            .open(&mut self.show_vert_data)
            .show(ui.ctx(), |ui| {
                if let Some(mesh) = mesh {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        mesh.draw_ui_vert_data(ui);
                    });
                }
            });

        egui::Window::new("Edge Data Flags")
            .open(&mut self.show_edge_data_flags)
            .show(ui.ctx(), |ui| {
                if let Some(mesh) = mesh {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        mesh.draw_ui_edge_flags(ui);
                    });
                }
            });

        egui::Window::new("Edge Data")
            .open(&mut self.show_edge_data)
            .show(ui.ctx(), |ui| {
                if let Some(mesh) = mesh {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        mesh.draw_ui_edge_data(ui);
                    });
                }
            });

        let mut show_computed_face_sizing = self.show_computed_face_sizing;
        egui::Window::new("Computed Face Sizing")
            .open(&mut show_computed_face_sizing)
            .show(ui.ctx(), |ui| {
                self.adaptive_remesh_params.draw_ui(&(), ui);
                ui.separator();
                if let Some(mesh) = mesh {
                    egui::ScrollArea::vertical().show(ui, |ui| match self.get_element() {
                        Element::Face => {
                            let face = mesh.get_faces().get_unknown_gen(self.get_element_index());
                            match face {
                                Some(face) => {
                                    let sizing = mesh.compute_dynamic_face_sizing(
                                        &self.adaptive_remesh_params,
                                        face.0,
                                    );
                                    match sizing {
                                        Ok(sizing) => {
                                            sizing.draw_ui(&(), ui);
                                        }
                                        Err(err) => {
                                            ui.label(format!("{}", err));
                                        }
                                    }
                                }
                                None => {
                                    ui.label(format!(
                                        "{}",
                                        MeshExtensionError::NoElementAtIndex(
                                            self.get_element_index()
                                        )
                                    ));
                                }
                            }
                        }
                        _ => {
                            ui.label("Face is not selected");
                        }
                    });
                } else {
                    ui.label("No Mesh Loaded");
                }
            });
        self.show_computed_face_sizing = show_computed_face_sizing;

        color_edit_button_dvec4(ui, "Fancy Node Color", &mut self.node_color);
        color_edit_button_dvec4(ui, "Fancy Vert Color", &mut self.vert_color);
        color_edit_button_dvec4(
            ui,
            "Fancy Node Vert Connect Color",
            &mut self.node_vert_connect_color,
        );
        color_edit_button_dvec4(ui, "Fancy Edge Color", &mut self.edge_color);
        color_edit_button_dvec4(ui, "Loose Fancy Node Color", &mut self.loose_node_color);
        color_edit_button_dvec4(ui, "Loose Fancy Vert Color", &mut self.loose_vert_color);
        color_edit_button_dvec4(ui, "Loose Fancy Edge Color", &mut self.loose_edge_color);
        color_edit_button_dvec4(
            ui,
            "Anisotropic Flippable Edge Fancy Edge Color",
            &mut self.anisotropic_flippable_edge_color,
        );
        color_edit_button_dvec4_range(ui, "Fancy Face Color Range", &mut self.face_color);
        color_edit_button_dvec4(ui, "Face Front Color", &mut self.face_front_color);
        color_edit_button_dvec4(ui, "Face Back Color", &mut self.face_back_color);
        color_edit_button_dvec4(ui, "Face Normal Color", &mut self.face_normal_color);
        color_edit_button_dvec4(ui, "Node Normal Color", &mut self.node_normal_color);
        color_edit_button_dvec4(
            ui,
            "Computed Node Normal Color",
            &mut self.computed_node_normal_color,
        );
        color_edit_button_dvec4(
            ui,
            "Face Violating Aspect Ratio Color",
            &mut self.face_violating_aspect_ratio_color,
        );
        color_edit_button_dvec4(
            ui,
            "Cloth Vertex Data Color",
            &mut self.cloth_vertex_data_color,
        );
        color_edit_button_dvec4(
            ui,
            "Edge Between Sewing Edges Color",
            &mut self.edge_between_sewing_edges_color,
        );
        color_edit_button_dvec4(ui, "Edge Data Flags Color", &mut self.edge_data_flags_color);

        ui.add(
            egui::Slider::new(&mut self.normal_pull_factor, 0.0..=3.0)
                .text("Bendiness of the fancy edges and faces"),
        );

        ui.add(egui::Slider::new(&mut self.face_normal_size, 0.0..=2.0).text("Face Normal Size"));
        ui.add(egui::Slider::new(&mut self.node_normal_size, 0.0..=2.0).text("Node Normal Size"));
        ui.add(
            egui::Slider::new(&mut self.computed_node_normal_size, 0.0..=2.0)
                .text("Computed Node Normal Size"),
        );

        ui.add(
            egui::Slider::new(&mut self.aspect_ratio_min, 0.0..=1.0).text("Aspect Ratio Minimum"),
        );

        // TODO clean up in rust 2021
        // https://doc.rust-lang.org/edition-guide/rust-2021/disjoint-capture-in-closures.html
        let adaptive_remesh_params = &mut self.adaptive_remesh_params;
        egui::Window::new("Adaptive Remesh Params")
            .open(&mut self.modify_adaptive_remesh_params)
            .show(ui.ctx(), |ui| {
                adaptive_remesh_params.draw_ui_edit(&(), ui);
            });

        egui::ComboBox::from_label("Aspect Ratio Metric")
            .selected_text(format!("{}", self.aspect_ratio_metric))
            .show_ui(ui, |ui| {
                AspectRatioMetric::all().for_each(|metric| {
                    ui.selectable_value(
                        &mut self.aspect_ratio_metric,
                        metric,
                        format!("{}", metric),
                    );
                });
            });

        egui::ComboBox::from_label("Cloth Vertex Elements")
            .selected_text(format!("{}", self.cloth_vertex_elements))
            .show_ui(ui, |ui| {
                ClothVertexElements::all().for_each(|element| {
                    ui.selectable_value(
                        &mut self.cloth_vertex_elements,
                        element,
                        format!("{}", element),
                    );
                });
            });

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

    #[allow(clippy::result_unit_err)]
    /// selects the element (based on which element type is currently
    /// selected) that intersects the closest with the given ray
    pub fn select_element(&mut self, ray: (glm::DVec3, glm::DVec3)) -> Result<(), ()> {
        match self.element {
            Element::Node => {
                let min_dist = 0.1;

                let mesh_3d_model_matrix = &self.mesh_transform.get_matrix();

                // find nearest within nodes
                let mut node_best = None;
                let mut node_best_dist = f64::MAX;
                for (node_index, node) in self
                    .get_mesh()
                    .map_err(|_| ())?
                    .as_ref()
                    .map_err(|_| ())?
                    .get_nodes()
                    .iter()
                {
                    let p1 = apply_model_matrix_vec3(&node.pos, mesh_3d_model_matrix);
                    let p1_to_ray_distance = glm::length(&glm::cross(&(p1 - ray.0), &ray.1));

                    if p1_to_ray_distance < min_dist
                        && (node_best.is_none() || node_best_dist > p1_to_ray_distance)
                    {
                        node_best = Some(node_index);
                        node_best_dist = p1_to_ray_distance;
                    }
                }

                if let Some(node_index) = node_best {
                    self.element_index = node_index.into_raw_parts().0;
                }

                Ok(())
            }
            Element::Vert => {
                let min_dist = 0.1;

                let uv_plane_3d_model_matrix = &self.uv_plane_3d_transform.get_matrix();

                // find nearest within verts
                let mut vert_best = None;
                let mut vert_best_dist = f64::MAX;
                for (vert_index, vert) in self
                    .get_mesh()
                    .map_err(|_| ())?
                    .as_ref()
                    .map_err(|_| ())?
                    .get_verts()
                    .iter()
                {
                    let p1 = apply_model_matrix_vec2(&vert.uv.unwrap(), uv_plane_3d_model_matrix);
                    let p1_to_ray_distance = glm::length(&glm::cross(&(p1 - ray.0), &ray.1));

                    if p1_to_ray_distance < min_dist
                        && (vert_best.is_none() || vert_best_dist > p1_to_ray_distance)
                    {
                        vert_best = Some(vert_index);
                        vert_best_dist = p1_to_ray_distance;
                    }
                }

                if let Some(vert_index) = vert_best {
                    self.element_index = vert_index.into_raw_parts().0;
                }

                Ok(())
            }
            Element::Edge => {
                let min_dist = 0.1;

                let uv_plane_3d_model_matrix = &self.uv_plane_3d_transform.get_matrix();
                let mesh_3d_model_matrix = &self.mesh_transform.get_matrix();

                let mut edge_best = None;
                let mut edge_best_ray_dist = f64::MAX;
                let mut edge_best_dist_to_edge = f64::MAX;

                let mesh = self.get_mesh().map_err(|_| ())?.as_ref().map_err(|_| ())?;

                mesh.get_edges().iter().for_each(|(edge_index, edge)| {
                    let uv1 = apply_model_matrix_vec2(
                        &mesh
                            .get_vert(edge.get_verts().unwrap().0)
                            .unwrap()
                            .uv
                            .unwrap(),
                        uv_plane_3d_model_matrix,
                    );
                    let uv2 = apply_model_matrix_vec2(
                        &mesh
                            .get_vert(edge.get_verts().unwrap().1)
                            .unwrap()
                            .uv
                            .unwrap(),
                        uv_plane_3d_model_matrix,
                    );

                    let edge_normal = glm::cross(&(uv1 - uv2), &ray.1).normalize();
                    let epos1 = uv1 + edge_normal * min_dist;
                    let epos2 = uv1 - edge_normal * min_dist;
                    let epos3 = uv2 + edge_normal * min_dist;
                    let epos4 = uv2 - edge_normal * min_dist;

                    if let Some((t, _, _)) =
                        ray_triangle_intersect(&ray.0, &ray.1, &epos1, &epos2, &epos4)
                    {
                        let dist_to_edge = glm::cross(&(uv1 - (ray.0 + t * ray.1)), &(uv2 - uv1))
                            .norm()
                            / (uv2 - uv1).norm();
                        if dist_to_edge < edge_best_dist_to_edge {
                            edge_best = Some(edge_index);
                            edge_best_ray_dist = t;
                            edge_best_dist_to_edge = dist_to_edge;
                        }
                    }

                    if let Some((t, _, _)) =
                        ray_triangle_intersect(&ray.0, &ray.1, &epos1, &epos4, &epos3)
                    {
                        let dist_to_edge = glm::cross(&(uv1 - (ray.0 + t * ray.1)), &(uv2 - uv1))
                            .norm()
                            / (uv2 - uv1).norm();
                        if dist_to_edge < edge_best_dist_to_edge {
                            edge_best = Some(edge_index);
                            edge_best_ray_dist = t;
                            edge_best_dist_to_edge = dist_to_edge;
                        }
                    }

                    // For the node edges (3d edges)
                    let pos1 = apply_model_matrix_vec3(
                        &mesh
                            .get_node(
                                mesh.get_vert(edge.get_verts().unwrap().0)
                                    .unwrap()
                                    .get_node()
                                    .unwrap(),
                            )
                            .unwrap()
                            .pos,
                        mesh_3d_model_matrix,
                    );
                    let pos2 = apply_model_matrix_vec3(
                        &mesh
                            .get_node(
                                mesh.get_vert(edge.get_verts().unwrap().1)
                                    .unwrap()
                                    .get_node()
                                    .unwrap(),
                            )
                            .unwrap()
                            .pos,
                        mesh_3d_model_matrix,
                    );

                    let edge_normal = glm::cross(&(pos1 - pos2), &ray.1).normalize();
                    let epos1 = pos1 + edge_normal * min_dist;
                    let epos2 = pos1 - edge_normal * min_dist;
                    let epos3 = pos2 + edge_normal * min_dist;
                    let epos4 = pos2 - edge_normal * min_dist;

                    if let Some((t, _, _)) =
                        ray_triangle_intersect(&ray.0, &ray.1, &epos1, &epos2, &epos4)
                    {
                        let dist_to_edge =
                            glm::cross(&(pos1 - (ray.0 + t * ray.1)), &(pos2 - pos1)).norm()
                                / (pos2 - pos1).norm();
                        if dist_to_edge < edge_best_dist_to_edge {
                            edge_best = Some(edge_index);
                            edge_best_ray_dist = t;
                            edge_best_dist_to_edge = dist_to_edge;
                        }
                    }

                    if let Some((t, _, _)) =
                        ray_triangle_intersect(&ray.0, &ray.1, &epos1, &epos4, &epos3)
                    {
                        let dist_to_edge =
                            glm::cross(&(pos1 - (ray.0 + t * ray.1)), &(pos2 - pos1)).norm()
                                / (pos2 - pos1).norm();
                        if dist_to_edge < edge_best_dist_to_edge {
                            edge_best = Some(edge_index);
                            edge_best_ray_dist = t;
                            edge_best_dist_to_edge = dist_to_edge;
                        }
                    }
                });

                if let Some(edge_index) = edge_best {
                    self.element_index = edge_index.into_raw_parts().0;
                }

                Ok(())
            }
            Element::Face => {
                let uv_plane_3d_model_matrix = &self.uv_plane_3d_transform.get_matrix();
                let mesh_3d_model_matrix = &self.mesh_transform.get_matrix();

                let mut face_best = None;
                let mut face_best_dist = f64::MAX;

                let mesh = self.get_mesh().map_err(|_| ())?.as_ref().map_err(|_| ())?;

                mesh.get_faces().iter().for_each(|(face_index, face)| {
                    assert_eq!(face.get_verts().len(), 3);

                    let uv_1 = apply_model_matrix_vec2(
                        &mesh.get_vert(face.get_verts()[0]).unwrap().uv.unwrap(),
                        uv_plane_3d_model_matrix,
                    );
                    let uv_2 = apply_model_matrix_vec2(
                        &mesh.get_vert(face.get_verts()[1]).unwrap().uv.unwrap(),
                        uv_plane_3d_model_matrix,
                    );
                    let uv_3 = apply_model_matrix_vec2(
                        &mesh.get_vert(face.get_verts()[2]).unwrap().uv.unwrap(),
                        uv_plane_3d_model_matrix,
                    );

                    if let Some((t, _, _)) =
                        ray_triangle_intersect(&ray.0, &ray.1, &uv_1, &uv_2, &uv_3)
                    {
                        if t < face_best_dist {
                            face_best = Some(face_index);
                            face_best_dist = t;
                        }
                    }

                    let pos_1 = apply_model_matrix_vec3(
                        &mesh
                            .get_node(
                                mesh.get_vert(face.get_verts()[0])
                                    .unwrap()
                                    .get_node()
                                    .unwrap(),
                            )
                            .unwrap()
                            .pos,
                        mesh_3d_model_matrix,
                    );
                    let pos_2 = apply_model_matrix_vec3(
                        &mesh
                            .get_node(
                                mesh.get_vert(face.get_verts()[1])
                                    .unwrap()
                                    .get_node()
                                    .unwrap(),
                            )
                            .unwrap()
                            .pos,
                        mesh_3d_model_matrix,
                    );
                    let pos_3 = apply_model_matrix_vec3(
                        &mesh
                            .get_node(
                                mesh.get_vert(face.get_verts()[2])
                                    .unwrap()
                                    .get_node()
                                    .unwrap(),
                            )
                            .unwrap()
                            .pos,
                        mesh_3d_model_matrix,
                    );

                    if let Some((t, _, _)) =
                        ray_triangle_intersect(&ray.0, &ray.1, &pos_1, &pos_2, &pos_3)
                    {
                        if t < face_best_dist {
                            face_best = Some(face_index);
                            face_best_dist = t;
                        }
                    }
                });

                if let Some(face_index) = face_best {
                    self.element_index = face_index.into_raw_parts().0;
                }

                Ok(())
            }
        }
    }

    pub fn get_mesh(&self) -> Result<&ResultMesh, MeshExtensionError> {
        self.meshes.get(self.mesh_index).map_or(
            Err(MeshExtensionError::NoElementAtIndex(self.mesh_index)),
            |loaded_mesh| Ok(loaded_mesh.get_mesh()),
        )
    }

    pub fn get_draw_mesh_with_shader(&self) -> MeshUseShader {
        self.draw_mesh_with_shader
    }

    pub fn get_draw_infinite_grid(&self) -> bool {
        self.draw_infinite_grid
    }

    pub fn get_draw_face_normals(&self) -> bool {
        self.draw_face_normals
    }

    pub fn get_draw_node_normals(&self) -> bool {
        self.draw_node_normals
    }

    pub fn get_draw_computed_node_normals(&self) -> bool {
        self.draw_computed_node_normals
    }

    pub fn get_draw_wireframe(&self) -> bool {
        self.draw_wireframe
    }

    pub fn get_draw_loose_nodes(&self) -> bool {
        self.draw_loose_nodes
    }

    pub fn get_draw_loose_verts(&self) -> bool {
        self.draw_loose_verts
    }

    pub fn get_draw_loose_edges(&self) -> bool {
        self.draw_loose_edges
    }

    pub fn get_draw_anisotropic_flippable_edges(&self) -> bool {
        self.draw_anisotropic_flippable_edges
    }

    pub fn get_draw_triangles_violating_aspect_ratio(&self) -> bool {
        self.draw_faces_violating_aspect_ratio
    }

    pub fn get_draw_cloth_vertex_data(&self) -> bool {
        self.draw_cloth_vertex_data
    }

    pub fn get_draw_edge_between_sewing_edges(&self) -> bool {
        self.draw_edge_between_sewing_edges
    }

    pub fn get_show_vert_data(&self) -> bool {
        self.show_vert_data
    }

    pub fn get_show_node_data(&self) -> bool {
        self.show_node_data
    }

    pub fn get_show_edge_data_flags(&self) -> bool {
        self.show_edge_data_flags
    }

    pub fn get_show_edge_data(&self) -> bool {
        self.show_edge_data
    }

    pub fn get_show_computed_face_sizing(&self) -> bool {
        self.show_computed_face_sizing
    }

    pub fn get_modify_adaptive_remesh_params(&self) -> bool {
        self.modify_adaptive_remesh_params
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

    pub fn get_loose_node_color(&self) -> glm::DVec4 {
        self.loose_node_color
    }

    pub fn get_loose_vert_color(&self) -> glm::DVec4 {
        self.loose_vert_color
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

    pub fn get_face_front_color(&self) -> glm::DVec4 {
        self.face_front_color
    }

    pub fn get_face_back_color(&self) -> glm::DVec4 {
        self.face_back_color
    }

    pub fn get_face_normal_color(&self) -> glm::DVec4 {
        self.face_normal_color
    }

    pub fn get_node_normal_color(&self) -> glm::DVec4 {
        self.node_normal_color
    }

    pub fn get_computed_node_normal_color(&self) -> glm::DVec4 {
        self.computed_node_normal_color
    }

    pub fn get_face_violating_aspect_ratio_color(&self) -> glm::DVec4 {
        self.face_violating_aspect_ratio_color
    }

    pub fn get_cloth_vertex_data_color(&self) -> glm::DVec4 {
        self.cloth_vertex_data_color
    }

    pub fn get_edge_between_sewing_edges_color(&self) -> glm::DVec4 {
        self.edge_between_sewing_edges_color
    }

    pub fn get_edge_data_flags_color(&self) -> glm::DVec4 {
        self.edge_data_flags_color
    }

    pub fn get_normal_pull_factor(&self) -> f64 {
        self.normal_pull_factor
    }

    pub fn get_face_normal_size(&self) -> f64 {
        self.face_normal_size
    }

    pub fn get_node_normal_size(&self) -> f64 {
        self.node_normal_size
    }

    pub fn get_computed_node_normal_size(&self) -> f64 {
        self.computed_node_normal_size
    }

    pub fn get_aspect_ratio_min(&self) -> f64 {
        self.aspect_ratio_min
    }

    pub fn get_adaptive_remesh_params(&self) -> &AdaptiveRemeshParams {
        &self.adaptive_remesh_params
    }

    pub fn get_aspect_ratio_metric(&self) -> AspectRatioMetric {
        self.aspect_ratio_metric
    }

    pub fn get_cloth_vertex_elements(&self) -> ClothVertexElements {
        self.cloth_vertex_elements
    }

    pub fn get_fps_limit(&self) -> f64 {
        self.fps_limit
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

/// From
/// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
pub fn ray_triangle_intersect(
    orig: &glm::DVec3,
    dir: &glm::DVec3,
    v0: &glm::DVec3,
    v1: &glm::DVec3,
    v2: &glm::DVec3,
) -> Option<(f64, f64, f64)> {
    let v0v1 = v1 - v0;
    let v0v2 = v2 - v0;
    let pvec = glm::cross(dir, &v0v2);

    let det = glm::dot(&v0v1, &pvec);

    if det.abs() < 0.001 {
        return None;
    }

    let inv_det = 1.0 / det;

    let tvec = orig - v0;
    let u = glm::dot(&tvec, &pvec) * inv_det;

    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let qvec = glm::cross(&tvec, &v0v1);
    let v = glm::dot(dir, &qvec) * inv_det;

    if v < 0.0 || (u + v) > 1.0 {
        return None;
    }

    let t = glm::dot(&v0v2, &qvec) * inv_det;

    Some((t, u, v))
}
