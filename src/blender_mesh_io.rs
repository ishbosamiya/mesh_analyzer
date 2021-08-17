use quick_renderer::egui::{self, Ui};
use quick_renderer::mesh::builtins::get_ico_sphere_subd_00;
use quick_renderer::mesh::{
    apply_model_matrix_to_normal, apply_model_matrix_vec2, apply_model_matrix_vec3, MeshDrawData,
};
use quick_renderer::shader::builtins::get_smooth_color_3d_shader;
use rmps::Deserializer;
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt::Display;
use std::io::BufRead;
use std::path::Path;

use quick_renderer::drawable::Drawable;
use quick_renderer::gpu_immediate::{GPUImmediate, GPUPrimType, GPUVertCompType, GPUVertFetchMode};
use quick_renderer::{glm, mesh, shader};

use crate::config::ClothVertexElements;
use crate::curve::{PointNormalTriangle, PointNormalTriangleDrawData};
use crate::prelude::DrawUI;
use crate::{config, util};
use crate::{
    config::Config,
    curve::{CubicBezierCurve, CubicBezierCurveDrawData},
};

use self::io_structs::ConversionError;

pub(crate) mod io_structs {
    use std::{collections::HashMap, convert::TryFrom, fmt::Display};

    use generational_arena::Arena;
    use quick_renderer::{glm, mesh};
    use serde::{Deserialize, Serialize};

    use crate::util;

    #[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Float3 {
        x: f32,
        y: f32,
        z: f32,
    }

    impl Display for Float3 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Float2 {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Float2x2 {
        m00: f32,
        m01: f32,
        m10: f32,
        m11: f32,
    }

    impl std::ops::Add for Float2x2 {
        type Output = Float2x2;

        fn add(self, rhs: Self) -> Self::Output {
            Self::Output {
                m00: self.m00 + rhs.m00,
                m01: self.m01 + rhs.m01,
                m10: self.m10 + rhs.m10,
                m11: self.m11 + rhs.m11,
            }
        }
    }

    impl std::ops::Mul<f32> for Float2x2 {
        type Output = Float2x2;

        fn mul(self, rhs: f32) -> Self::Output {
            Self::Output {
                m00: self.m00 * rhs,
                m01: self.m01 * rhs,
                m10: self.m10 * rhs,
                m11: self.m11 * rhs,
            }
        }
    }

    impl From<Float2x2> for glm::Mat2 {
        fn from(f: Float2x2) -> Self {
            glm::mat2(f.m00, f.m01, f.m10, f.m11)
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Index {
        index: usize,
        generation: usize,
    }

    impl Index {
        pub fn get_index(&self) -> usize {
            self.index
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ClothVertex {
        pub flags: i32,
        pub v: Float3,
        pub xconst: Float3,
        pub x: Float3,
        pub xold: Float3,
        pub tx: Float3,
        pub txold: Float3,
        pub tv: Float3,
        pub mass: f32,
        pub goal: f32,
        pub impulse: Float3,
        pub xrest: Float3,
        pub dcvel: Float3,
        pub impulse_count: u32,
        pub avg_spring_len: f32,
        pub struct_stiff: f32,
        pub bend_stiff: f32,
        pub shear_stiff: f32,
        pub spring_count: i32,
        pub shrink_factor: f32,
        pub internal_stiff: f32,
        pub pressure_factor: f32,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ClothNodeData {
        cloth_node_data_str: String,
        cloth_node_data: ClothVertex,
    }

    impl ClothNodeData {
        pub fn get_cloth_node_data(&self) -> &ClothVertex {
            &self.cloth_node_data
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct NodeData<T> {
        node_data_str: String,
        extra_data: T,
    }

    impl<T> NodeData<T> {
        pub fn get_extra_data(&self) -> &T {
            &self.extra_data
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Sizing {
        sizing_str: String,
        m: Float2x2,
    }

    impl Sizing {
        pub fn new(m: Float2x2) -> Self {
            Self {
                sizing_str: String::new(),
                m,
            }
        }

        pub fn get_m(&self) -> &Float2x2 {
            &self.m
        }

        pub fn get_edge_size_sq(&self, other: &Sizing, u_i: &glm::DVec2, u_j: &glm::DVec2) -> f32 {
            let u_ij: glm::Vec2 = glm::convert(u_i - u_j);

            let m_avg: glm::Mat2 = ((self.m + other.m) * 0.5).into();

            glm::dot(&u_ij, &(m_avg * u_ij))
        }
    }

    impl std::ops::Add for &Sizing {
        type Output = Sizing;

        fn add(self, rhs: Self) -> Self::Output {
            Sizing::new(self.m + rhs.m)
        }
    }

    impl std::ops::Mul<f32> for &Sizing {
        type Output = Sizing;

        fn mul(self, rhs: f32) -> Self::Output {
            Sizing::new(self.m * rhs)
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct VertData {
        vert_data_str: String,
        sizing: Sizing,
        flag: i32,
    }

    impl VertData {
        pub fn get_sizing(&self) -> &Sizing {
            &self.sizing
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EdgeData {
        edge_data_str: String,
        size: f32,
    }

    impl EdgeData {
        pub fn get_size(&self) -> f32 {
            self.size
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct EmptyExtraData {
        empty_extra_data_str: String,
    }

    pub type NodeIndex = Index;
    pub type VertIndex = Index;
    pub type EdgeIndex = Index;
    pub type FaceIndex = Index;

    pub type IncidentVerts = Vec<VertIndex>;
    pub type IncidentEdges = Vec<EdgeIndex>;
    pub type IncidentFaces = Vec<FaceIndex>;
    pub type AdjacentVerts = IncidentVerts;
    pub type EdgeVerts = (VertIndex, VertIndex);

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Node<T> {
        self_index: NodeIndex,
        verts: IncidentVerts,

        pos: Float3,
        normal: Float3,
        extra_data: Option<T>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Vert<T> {
        self_index: VertIndex,
        edges: IncidentEdges,
        node: Option<NodeIndex>,

        uv: Float2,
        extra_data: Option<T>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Edge<T> {
        self_index: EdgeIndex,
        faces: IncidentFaces,
        verts: Option<EdgeVerts>,

        extra_data: Option<T>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Face<T> {
        self_index: FaceIndex,
        verts: AdjacentVerts,

        normal: Float3,
        extra_data: Option<T>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Mesh<END, EVD, EED, EFD> {
        pub nodes_str: String,
        pub nodes: Vec<Node<END>>,
        pub verts_str: String,
        pub verts: Vec<Vert<EVD>>,
        pub edges_str: String,
        pub edges: Vec<Edge<EED>>,
        pub faces_str: String,
        pub faces: Vec<Face<EFD>>,

        pub nodes_map_str: String,
        pub node_pos_index_map: HashMap<NodeIndex, usize>,
        pub verts_map_str: String,
        pub vert_pos_index_map: HashMap<VertIndex, usize>,
        pub edges_map_str: String,
        pub edge_pos_index_map: HashMap<EdgeIndex, usize>,
        pub faces_map_str: String,
        pub face_pos_index_map: HashMap<FaceIndex, usize>,
    }

    impl From<Float3> for glm::DVec3 {
        fn from(float3: Float3) -> Self {
            glm::vec3(float3.x.into(), float3.y.into(), float3.z.into())
        }
    }

    impl From<Float2> for glm::DVec2 {
        fn from(float2: Float2) -> Self {
            glm::vec2(float2.x.into(), float2.y.into())
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum ConversionError {
        NoElementInPosIndexMap(Index),
    }

    impl Display for ConversionError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ConversionError::NoElementInPosIndexMap(index) => {
                    write!(f, "No element in pos index map: {:?}", index)
                }
            }
        }
    }

    impl std::error::Error for ConversionError {}

    impl<END, EVD, EED, EFD> TryFrom<Mesh<END, EVD, EED, EFD>> for mesh::Mesh<END, EVD, EED, EFD> {
        type Error = ConversionError;

        fn try_from(mut io_mesh: Mesh<END, EVD, EED, EFD>) -> Result<Self, Self::Error> {
            // The referencing between the elements is set after all
            // the elements are added to their respective arenas so
            // that the positional mapping can be used to the correct
            // arena index of the elements

            let mut nodes = Arena::new();
            let mut verts = Arena::new();
            let mut edges = Arena::new();
            let mut faces = Arena::new();

            io_mesh.nodes.iter_mut().for_each(|io_node| {
                nodes.insert_with(|self_index| {
                    let mut node = mesh::Node::new(mesh::NodeIndex(self_index), io_node.pos.into());
                    node.pos = io_node.pos.into();
                    node.normal = Some(io_node.normal.into());
                    node.extra_data = io_node.extra_data.take();

                    node
                });
            });

            io_mesh.verts.iter_mut().for_each(|io_vert| {
                verts.insert_with(|self_index| {
                    let mut vert = mesh::Vert::new(mesh::VertIndex(self_index));

                    vert.uv = Some(io_vert.uv.into());
                    vert.extra_data = io_vert.extra_data.take();

                    vert
                });
            });

            io_mesh.edges.iter_mut().for_each(|io_edge| {
                edges.insert_with(|self_index| {
                    let mut edge = mesh::Edge::new(mesh::EdgeIndex(self_index));

                    edge.extra_data = io_edge.extra_data.take();

                    edge
                });
            });

            io_mesh.faces.iter_mut().for_each(|io_face| {
                faces.insert_with(|self_index| {
                    let mut face = mesh::Face::new(mesh::FaceIndex(self_index));

                    face.normal = Some(io_face.normal.into());
                    face.extra_data = io_face.extra_data.take();

                    face
                });
            });

            assert_eq!(io_mesh.faces.len(), faces.len());
            assert_eq!(io_mesh.edges.len(), edges.len());
            assert_eq!(io_mesh.verts.len(), verts.len());
            assert_eq!(io_mesh.nodes.len(), nodes.len());

            nodes
                .iter_mut()
                .zip(io_mesh.nodes.iter())
                .try_for_each(|(node, io_node)| {
                    let node_verts;
                    unsafe {
                        node_verts = node.1.get_verts_mut();
                    }
                    io_node.verts.iter().try_for_each(|io_node_vert_index| {
                        let vert_i = io_mesh
                            .vert_pos_index_map
                            .get(io_node_vert_index)
                            .ok_or(ConversionError::NoElementInPosIndexMap(*io_node_vert_index))?;
                        node_verts.push(mesh::VertIndex(verts.get_unknown_gen(*vert_i).unwrap().1));
                        Ok(())
                    })?;
                    Ok(())
                })?;

            verts
                .iter_mut()
                .zip(io_mesh.verts.iter())
                .try_for_each(|(vert, io_vert)| {
                    let vert_edges;
                    unsafe {
                        vert_edges = vert.1.get_edges_mut();
                    }
                    io_vert.edges.iter().try_for_each(|io_vert_edge_index| {
                        let edge_i = io_mesh
                            .edge_pos_index_map
                            .get(io_vert_edge_index)
                            .ok_or(ConversionError::NoElementInPosIndexMap(*io_vert_edge_index))?;
                        vert_edges.push(mesh::EdgeIndex(edges.get_unknown_gen(*edge_i).unwrap().1));
                        Ok(())
                    })?;

                    let vert_node;
                    unsafe {
                        vert_node = vert.1.get_node_mut();
                    }

                    *vert_node = io_vert.node.as_ref().map_or_else(
                        || Ok(None),
                        |io_vert_node_index| {
                            let node_i = io_mesh.node_pos_index_map.get(io_vert_node_index).ok_or(
                                ConversionError::NoElementInPosIndexMap(*io_vert_node_index),
                            )?;
                            Ok(Some(mesh::NodeIndex(
                                nodes.get_unknown_gen(*node_i).unwrap().1,
                            )))
                        },
                    )?;

                    Ok(())
                })?;

            edges
                .iter_mut()
                .zip(io_mesh.edges.iter())
                .try_for_each(|(edge, io_edge)| {
                    let edge_verts;
                    unsafe {
                        edge_verts = edge.1.get_verts_mut();
                    }
                    *edge_verts = io_edge.verts.map_or_else(
                        || Ok(None),
                        |(io_edge_vert_1_index, io_edge_vert_2_index)| {
                            let edge_vert_1_i = io_mesh
                                .vert_pos_index_map
                                .get(&io_edge_vert_1_index)
                                .ok_or(ConversionError::NoElementInPosIndexMap(
                                    io_edge_vert_1_index,
                                ))?;

                            let edge_vert_2_i = io_mesh
                                .vert_pos_index_map
                                .get(&io_edge_vert_2_index)
                                .ok_or(ConversionError::NoElementInPosIndexMap(
                                    io_edge_vert_2_index,
                                ))?;

                            Ok(Some((
                                mesh::VertIndex(verts.get_unknown_gen(*edge_vert_1_i).unwrap().1),
                                mesh::VertIndex(verts.get_unknown_gen(*edge_vert_2_i).unwrap().1),
                            )))
                        },
                    )?;

                    let edge_faces;
                    unsafe {
                        edge_faces = edge.1.get_faces_mut();
                    }
                    io_edge.faces.iter().try_for_each(|io_edge_face_index| {
                        let face_i = io_mesh
                            .face_pos_index_map
                            .get(io_edge_face_index)
                            .ok_or(ConversionError::NoElementInPosIndexMap(*io_edge_face_index))?;
                        edge_faces.push(mesh::FaceIndex(faces.get_unknown_gen(*face_i).unwrap().1));
                        Ok(())
                    })?;

                    Ok(())
                })?;

            faces
                .iter_mut()
                .zip(io_mesh.faces.iter())
                .try_for_each(|(face, io_face)| {
                    let face_verts;
                    unsafe {
                        face_verts = face.1.get_verts_mut();
                    }
                    io_face.verts.iter().try_for_each(|io_face_vert_index| {
                        let vert_i = io_mesh
                            .vert_pos_index_map
                            .get(io_face_vert_index)
                            .ok_or(ConversionError::NoElementInPosIndexMap(*io_face_vert_index))?;
                        face_verts.push(mesh::VertIndex(verts.get_unknown_gen(*vert_i).unwrap().1));
                        Ok(())
                    })?;
                    Ok(())
                })?;

            let mut mesh = Self::from_arenas(nodes, verts, edges, faces);
            // since the mesh is from Blender, we need to convert the
            // mesh coordinates from Blender's to OpenGL
            mesh.apply_model_matrix(&util::axis_conversion_matrix(
                util::Axis::Y,
                util::Axis::Z,
                util::Axis::NegZ,
                util::Axis::Y,
            ));
            Ok(mesh)
        }
    }

    // pub type AdaptiveNode<END> = Node<NodeData<END>>;
    // pub type AdaptiveVert = Vert<VertData>;
    // pub type AdaptiveEdge = Edge<EdgeData>;
    // pub type AdaptiveFace = Face<EmptyExtraData>;
    // pub type AdaptiveMesh<END> = Mesh<NodeData<END>, VertData, EdgeData, EmptyExtraData>;
    // pub type EmptyAdaptiveMesh = AdaptiveMesh<EmptyExtraData>;
}

pub type AdaptiveNode<END> = mesh::Node<io_structs::NodeData<END>>;
pub type AdaptiveVert = mesh::Vert<io_structs::VertData>;
pub type AdaptiveEdge = mesh::Edge<io_structs::EdgeData>;
pub type AdaptiveFace = mesh::Face<io_structs::EmptyExtraData>;
pub type AdaptiveMesh<END> = mesh::Mesh<
    io_structs::NodeData<END>,
    io_structs::VertData,
    io_structs::EdgeData,
    io_structs::EmptyExtraData,
>;
pub type EmptyAdaptiveMesh = AdaptiveMesh<io_structs::EmptyExtraData>;
pub type ClothAdaptiveMesh = AdaptiveMesh<io_structs::ClothNodeData>;

pub trait AdaptiveMeshExtension<END> {
    fn is_edge_flippable_anisotropic_aware(
        &self,
        edge: &AdaptiveEdge,
    ) -> Result<bool, MeshExtensionError>;

    fn compute_edge_size_sq_from_v1_v2(
        &self,
        v1: &AdaptiveVert,
        v2: &AdaptiveVert,
    ) -> Result<f32, MeshExtensionError>;

    fn compute_edge_size_sq_from_edge(
        &self,
        edge: &AdaptiveEdge,
    ) -> Result<f32, MeshExtensionError>;

    fn adaptive_mesh_visualize_config(
        &self,
        config: &Config<
            io_structs::NodeData<END>,
            io_structs::VertData,
            io_structs::EdgeData,
            io_structs::EmptyExtraData,
        >,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError>;
}

impl<END> AdaptiveMeshExtension<END> for AdaptiveMesh<END> {
    fn is_edge_flippable_anisotropic_aware(
        &self,
        edge: &AdaptiveEdge,
    ) -> Result<bool, MeshExtensionError> {
        let alpha = 0.1;

        if self.is_edge_loose_or_on_seam_or_boundary(edge) {
            return Ok(false);
        }

        if !self.is_edge_flippable(edge, false) {
            return Ok(false);
        }

        // if flipping the edge causes the edge size metric to not be under 1
        {
            let ov1_index =
                self.get_checked_other_vert_index(edge.get_self_index(), edge.get_faces()[0]);
            let ov2_index =
                self.get_checked_other_vert_index(edge.get_self_index(), edge.get_faces()[1]);

            let ov1 = self.get_vert(ov1_index).unwrap();
            let ov2 = self.get_vert(ov2_index).unwrap();

            if self.compute_edge_size_sq_from_v1_v2(ov1, ov2)? > 1.0 {
                return Ok(false);
            }
        }

        let cross_2d = |a: &glm::DVec2, b: &glm::DVec2| a[0] * b[1] - a[1] * b[0];

        let (v_i_index, v_j_index) = edge.get_verts().unwrap();
        let v_k_index =
            self.get_checked_other_vert_index(edge.get_self_index(), edge.get_faces()[0]);
        let v_l_index =
            self.get_checked_other_vert_index(edge.get_self_index(), edge.get_faces()[1]);
        let v_i = self.get_vert(v_i_index).unwrap();
        let v_j = self.get_vert(v_j_index).unwrap();
        let v_k = self.get_vert(v_k_index).unwrap();
        let v_l = self.get_vert(v_l_index).unwrap();

        let m_i = v_i
            .extra_data
            .as_ref()
            .ok_or(MeshExtensionError::NoExtraData)?
            .get_sizing();
        let m_j = v_j
            .extra_data
            .as_ref()
            .ok_or(MeshExtensionError::NoExtraData)?
            .get_sizing();
        let m_k = v_k
            .extra_data
            .as_ref()
            .ok_or(MeshExtensionError::NoExtraData)?
            .get_sizing();
        let m_l = v_l
            .extra_data
            .as_ref()
            .ok_or(MeshExtensionError::NoExtraData)?
            .get_sizing();

        let m_avg = &(&(m_i + m_j) + &(m_k + m_l)) * 0.25;
        let m_avg = *m_avg.get_m();
        let m_avg: glm::DMat2 = glm::convert::<glm::Mat2, _>(m_avg.into());

        let u_jk = v_j.uv.unwrap() - v_k.uv.unwrap();
        let u_ik = v_i.uv.unwrap() - v_k.uv.unwrap();
        let u_il = v_i.uv.unwrap() - v_l.uv.unwrap();
        let u_jl = v_j.uv.unwrap() - v_l.uv.unwrap();

        let lhs = cross_2d(&u_jk, &u_ik) * glm::dot(&u_il, &(m_avg * u_jl))
            + glm::dot(&u_jk, &(m_avg * u_ik)) * cross_2d(&u_il, &u_jl);

        let rhs = -alpha * (cross_2d(&u_jk, &u_ik).abs() + cross_2d(&u_il, &u_jl).abs());

        Ok(lhs < rhs)
    }

    fn adaptive_mesh_visualize_config(
        &self,
        config: &Config<
            io_structs::NodeData<END>,
            io_structs::VertData,
            io_structs::EdgeData,
            io_structs::EmptyExtraData,
        >,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError> {
        let mut had_extra_data = true;
        if config.get_draw_anisotropic_flippable_edges() {
            let uv_plane_3d_model_matrix = config.get_uv_plane_3d_transform().get_matrix();
            self.get_edges()
                .iter()
                .filter(|(_, edge)| {
                    self.is_edge_flippable_anisotropic_aware(edge)
                        .unwrap_or_else(|_| {
                            had_extra_data = false;
                            false
                        })
                })
                .for_each(|(_, edge)| {
                    self.draw_fancy_edge(
                        edge,
                        &uv_plane_3d_model_matrix,
                        imm,
                        glm::convert(config.get_anisotropic_flippable_edge_color()),
                        config.get_normal_pull_factor(),
                    );
                });
        }

        if had_extra_data {
            Ok(())
        } else {
            Err(MeshExtensionError::NoExtraData)
        }
    }

    fn compute_edge_size_sq_from_v1_v2(
        &self,
        v1: &AdaptiveVert,
        v2: &AdaptiveVert,
    ) -> Result<f32, MeshExtensionError> {
        let v1_uv = v1.uv.unwrap();
        let v2_uv = v2.uv.unwrap();
        let v1_sizing = v1
            .extra_data
            .as_ref()
            .ok_or(MeshExtensionError::NoExtraData)?
            .get_sizing();
        let v2_sizing = v2
            .extra_data
            .as_ref()
            .ok_or(MeshExtensionError::NoExtraData)?
            .get_sizing();

        Ok(v1_sizing.get_edge_size_sq(v2_sizing, &v1_uv, &v2_uv))
    }

    fn compute_edge_size_sq_from_edge(
        &self,
        edge: &AdaptiveEdge,
    ) -> Result<f32, MeshExtensionError> {
        let (v1, v2) = self.get_checked_verts_of_edge(edge, false);

        self.compute_edge_size_sq_from_v1_v2(v1, v2)
    }
}

pub trait ClothAdaptiveMeshExtension {
    fn cloth_adaptive_mesh_visualize_config(
        &self,
        config: &Config<
            io_structs::NodeData<io_structs::ClothNodeData>,
            io_structs::VertData,
            io_structs::EdgeData,
            io_structs::EmptyExtraData,
        >,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError>;
}

impl ClothAdaptiveMeshExtension for ClothAdaptiveMesh {
    fn cloth_adaptive_mesh_visualize_config(
        &self,
        config: &Config<
            io_structs::NodeData<io_structs::ClothNodeData>,
            io_structs::VertData,
            io_structs::EdgeData,
            io_structs::EmptyExtraData,
        >,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError> {
        if config.get_draw_cloth_vertex_data() {
            enum DrawStyle {
                None,
                Point,
                Line,
            }

            let mut draw_style = DrawStyle::None;
            let mut points: Vec<glm::DVec3> = Vec::new();

            match config.get_cloth_vertex_elements() {
                ClothVertexElements::Flags => {}
                ClothVertexElements::V => self
                    .get_nodes()
                    .iter()
                    .try_for_each::<_, Result<(), MeshExtensionError>>(|(_, node)| {
                        let cloth_vertex = node
                            .extra_data
                            .as_ref()
                            .ok_or(MeshExtensionError::NoExtraData)?
                            .get_extra_data()
                            .get_cloth_node_data();
                        let x = cloth_vertex.x.into();
                        let v: glm::DVec3 = cloth_vertex.v.into();
                        points.push(x);
                        points.push(x + v);
                        draw_style = DrawStyle::Line;
                        Ok(())
                    })?,
                ClothVertexElements::Xconst => {
                    self.get_nodes()
                        .iter()
                        .try_for_each::<_, Result<(), MeshExtensionError>>(|(_, node)| {
                            let cloth_vertex = node
                                .extra_data
                                .as_ref()
                                .ok_or(MeshExtensionError::NoExtraData)?
                                .get_extra_data()
                                .get_cloth_node_data();
                            let xconst = cloth_vertex.xconst.into();
                            points.push(xconst);
                            draw_style = DrawStyle::Point;
                            Ok(())
                        })?
                }
                ClothVertexElements::X => self
                    .get_nodes()
                    .iter()
                    .try_for_each::<_, Result<(), MeshExtensionError>>(|(_, node)| {
                        let cloth_vertex = node
                            .extra_data
                            .as_ref()
                            .ok_or(MeshExtensionError::NoExtraData)?
                            .get_extra_data()
                            .get_cloth_node_data();
                        let x = cloth_vertex.x.into();
                        points.push(x);
                        draw_style = DrawStyle::Point;
                        Ok(())
                    })?,
                ClothVertexElements::Xold => {
                    self.get_nodes()
                        .iter()
                        .try_for_each::<_, Result<(), MeshExtensionError>>(|(_, node)| {
                            let cloth_vertex = node
                                .extra_data
                                .as_ref()
                                .ok_or(MeshExtensionError::NoExtraData)?
                                .get_extra_data()
                                .get_cloth_node_data();
                            let xold = cloth_vertex.xold.into();
                            points.push(xold);
                            draw_style = DrawStyle::Point;
                            Ok(())
                        })?
                }
                ClothVertexElements::Tx => self
                    .get_nodes()
                    .iter()
                    .try_for_each::<_, Result<(), MeshExtensionError>>(|(_, node)| {
                        let cloth_vertex = node
                            .extra_data
                            .as_ref()
                            .ok_or(MeshExtensionError::NoExtraData)?
                            .get_extra_data()
                            .get_cloth_node_data();
                        let tx = cloth_vertex.tx.into();
                        points.push(tx);
                        draw_style = DrawStyle::Point;
                        Ok(())
                    })?,
                ClothVertexElements::Txold => {
                    self.get_nodes()
                        .iter()
                        .try_for_each::<_, Result<(), MeshExtensionError>>(|(_, node)| {
                            let cloth_vertex = node
                                .extra_data
                                .as_ref()
                                .ok_or(MeshExtensionError::NoExtraData)?
                                .get_extra_data()
                                .get_cloth_node_data();
                            let txold = cloth_vertex.txold.into();
                            points.push(txold);
                            draw_style = DrawStyle::Point;
                            Ok(())
                        })?
                }
                ClothVertexElements::Tv => self
                    .get_nodes()
                    .iter()
                    .try_for_each::<_, Result<(), MeshExtensionError>>(|(_, node)| {
                        let cloth_vertex = node
                            .extra_data
                            .as_ref()
                            .ok_or(MeshExtensionError::NoExtraData)?
                            .get_extra_data()
                            .get_cloth_node_data();
                        let x = cloth_vertex.x.into();
                        let tv: glm::DVec3 = cloth_vertex.tv.into();
                        points.push(x);
                        points.push(x + tv);
                        draw_style = DrawStyle::Line;
                        Ok(())
                    })?,
                ClothVertexElements::Mass => {}
                ClothVertexElements::Goal => {}
                ClothVertexElements::Impulse => {
                    self.get_nodes()
                        .iter()
                        .try_for_each::<_, Result<(), MeshExtensionError>>(|(_, node)| {
                            let cloth_vertex = node
                                .extra_data
                                .as_ref()
                                .ok_or(MeshExtensionError::NoExtraData)?
                                .get_extra_data()
                                .get_cloth_node_data();
                            let x = cloth_vertex.x.into();
                            let impulse: glm::DVec3 = cloth_vertex.impulse.into();
                            points.push(x);
                            points.push(x + impulse);
                            draw_style = DrawStyle::Line;
                            Ok(())
                        })?
                }
                ClothVertexElements::Xrest => {
                    self.get_nodes()
                        .iter()
                        .try_for_each::<_, Result<(), MeshExtensionError>>(|(_, node)| {
                            let cloth_vertex = node
                                .extra_data
                                .as_ref()
                                .ok_or(MeshExtensionError::NoExtraData)?
                                .get_extra_data()
                                .get_cloth_node_data();
                            let xrest = cloth_vertex.xrest.into();
                            points.push(xrest);
                            draw_style = DrawStyle::Point;
                            Ok(())
                        })?
                }
                ClothVertexElements::Dcvel => {
                    self.get_nodes()
                        .iter()
                        .try_for_each::<_, Result<(), MeshExtensionError>>(|(_, node)| {
                            let cloth_vertex = node
                                .extra_data
                                .as_ref()
                                .ok_or(MeshExtensionError::NoExtraData)?
                                .get_extra_data()
                                .get_cloth_node_data();
                            let x = cloth_vertex.x.into();
                            let dcvel: glm::DVec3 = cloth_vertex.dcvel.into();
                            points.push(x);
                            points.push(x + dcvel);
                            draw_style = DrawStyle::Line;
                            Ok(())
                        })?
                }
                ClothVertexElements::ImpulseCount => {}
                ClothVertexElements::AvgSpringLen => {}
                ClothVertexElements::StructStiff => {}
                ClothVertexElements::BendStiff => {}
                ClothVertexElements::ShearStiff => {}
                ClothVertexElements::SpringCount => {}
                ClothVertexElements::ShrinkFactor => {}
                ClothVertexElements::InternalStiff => {}
                ClothVertexElements::PressureFactor => {}
            }

            let color = glm::convert(config.get_cloth_vertex_data_color());
            let axis_conversion_matrix = &util::axis_conversion_matrix(
                util::Axis::Y,
                util::Axis::Z,
                util::Axis::NegZ,
                util::Axis::Y,
            );
            let mesh_3d_model_matrix = &config.get_mesh_transform().get_matrix();

            match draw_style {
                DrawStyle::Point => {
                    points.iter().for_each(|pos| {
                        let pos = apply_model_matrix_vec3(pos, axis_conversion_matrix);
                        let pos = apply_model_matrix_vec3(&pos, mesh_3d_model_matrix);
                        draw_sphere_at(&pos, color, imm);
                    });
                }
                DrawStyle::Line => {
                    let positions: Vec<_> = points
                        .iter()
                        .map(|pos| {
                            let pos = apply_model_matrix_vec3(pos, axis_conversion_matrix);
                            apply_model_matrix_vec3(&pos, mesh_3d_model_matrix)
                        })
                        .collect();
                    draw_lines(&positions, color, imm);
                }
                DrawStyle::None => {}
            }

            Ok(())
        } else {
            Ok(())
        }
    }
}

impl DrawUI for ClothAdaptiveMesh {
    type ExtraData = Config<
        io_structs::NodeData<io_structs::ClothNodeData>,
        io_structs::VertData,
        io_structs::EdgeData,
        io_structs::EmptyExtraData,
    >;

    fn draw_ui(&self, extra_data: &Self::ExtraData, ui: &mut Ui) {
        let config = &extra_data;
        if config.get_draw_cloth_vertex_data() {
            egui::Window::new("Cloth Vertex Data").show(ui.ctx(), |ui| {
                egui::ScrollArea::auto_sized().show(ui, |ui| {
                    self.get_nodes()
                        .iter()
                        .enumerate()
                        .for_each(|(i, (_, node))| {
                            ui.horizontal(|ui| {
                                ui.label(format!("{:04}", i));
                                let cloth_vertex = node.extra_data.as_ref().map(|node_data| {
                                    node_data.get_extra_data().get_cloth_node_data()
                                });
                                if let Some(cloth_vertex) = cloth_vertex {
                                    match config.get_cloth_vertex_elements() {
                                        ClothVertexElements::Flags => {
                                            ui.label(format!("{}", cloth_vertex.flags))
                                        }
                                        ClothVertexElements::V => {
                                            ui.label(format!("{}", cloth_vertex.v))
                                        }
                                        ClothVertexElements::Xconst => {
                                            ui.label(format!("{}", cloth_vertex.xconst))
                                        }
                                        ClothVertexElements::X => {
                                            ui.label(format!("{}", cloth_vertex.x))
                                        }
                                        ClothVertexElements::Xold => {
                                            ui.label(format!("{}", cloth_vertex.xold))
                                        }
                                        ClothVertexElements::Tx => {
                                            ui.label(format!("{}", cloth_vertex.tx))
                                        }
                                        ClothVertexElements::Txold => {
                                            ui.label(format!("{}", cloth_vertex.txold))
                                        }
                                        ClothVertexElements::Tv => {
                                            ui.label(format!("{}", cloth_vertex.tv))
                                        }
                                        ClothVertexElements::Mass => {
                                            ui.label(format!("{}", cloth_vertex.mass))
                                        }
                                        ClothVertexElements::Goal => {
                                            ui.label(format!("{}", cloth_vertex.goal))
                                        }
                                        ClothVertexElements::Impulse => {
                                            ui.label(format!("{}", cloth_vertex.impulse))
                                        }
                                        ClothVertexElements::Xrest => {
                                            ui.label(format!("{}", cloth_vertex.xrest))
                                        }
                                        ClothVertexElements::Dcvel => {
                                            ui.label(format!("{}", cloth_vertex.dcvel))
                                        }
                                        ClothVertexElements::ImpulseCount => {
                                            ui.label(format!("{}", cloth_vertex.impulse_count))
                                        }
                                        ClothVertexElements::AvgSpringLen => {
                                            ui.label(format!("{}", cloth_vertex.avg_spring_len))
                                        }
                                        ClothVertexElements::StructStiff => {
                                            ui.label(format!("{}", cloth_vertex.struct_stiff))
                                        }
                                        ClothVertexElements::BendStiff => {
                                            ui.label(format!("{}", cloth_vertex.bend_stiff))
                                        }
                                        ClothVertexElements::ShearStiff => {
                                            ui.label(format!("{}", cloth_vertex.shear_stiff))
                                        }
                                        ClothVertexElements::SpringCount => {
                                            ui.label(format!("{}", cloth_vertex.spring_count))
                                        }
                                        ClothVertexElements::ShrinkFactor => {
                                            ui.label(format!("{}", cloth_vertex.shrink_factor))
                                        }
                                        ClothVertexElements::InternalStiff => {
                                            ui.label(format!("{}", cloth_vertex.internal_stiff))
                                        }
                                        ClothVertexElements::PressureFactor => {
                                            ui.label(format!("{}", cloth_vertex.pressure_factor))
                                        }
                                    };
                                }
                            });
                        });
                });
            });
        }
    }

    fn draw_ui_edit(&mut self, _extra_data: &Self::ExtraData, _ui: &mut Ui) {
        unreachable!("No operation needs this yet")
    }
}

pub struct MeshUVDrawData<'a> {
    imm: &'a mut GPUImmediate,
    uv_plane_3d_model_matrix: &'a glm::Mat4,
    color: &'a glm::Vec4,
}

impl<'a> MeshUVDrawData<'a> {
    pub fn new(
        imm: &'a mut GPUImmediate,
        uv_plane_3d_model_matrix: &'a glm::Mat4,
        color: &'a glm::Vec4,
    ) -> Self {
        Self {
            imm,
            uv_plane_3d_model_matrix,
            color,
        }
    }
}

pub struct MeshElementReferences {
    pub nodes: Option<Vec<mesh::NodeIndex>>,
    pub verts: Option<Vec<mesh::VertIndex>>,
    pub edges: Option<Vec<mesh::EdgeIndex>>,
    pub faces: Option<Vec<mesh::FaceIndex>>,
}

impl MeshElementReferences {
    pub fn new(
        nodes: Vec<mesh::NodeIndex>,
        verts: Vec<mesh::VertIndex>,
        edges: Vec<mesh::EdgeIndex>,
        faces: Vec<mesh::FaceIndex>,
    ) -> Self {
        Self {
            nodes: Some(nodes),
            verts: Some(verts),
            edges: Some(edges),
            faces: Some(faces),
        }
    }

    pub fn from_node<T>(node: &mesh::Node<T>) -> Self {
        Self {
            nodes: None,
            verts: (!node.get_verts().is_empty()).then(|| node.get_verts().clone()),
            edges: None,
            faces: None,
        }
    }

    pub fn from_vert<T>(vert: &mesh::Vert<T>) -> Self {
        Self {
            nodes: vert.get_node().map(|node_index| vec![node_index]),
            verts: None,
            edges: (!vert.get_edges().is_empty()).then(|| vert.get_edges().clone()),
            faces: None,
        }
    }

    pub fn from_edge<T>(edge: &mesh::Edge<T>) -> Self {
        Self {
            nodes: None,
            verts: edge
                .get_verts()
                .map(|(vert_index_1, vert_index_2)| vec![vert_index_1, vert_index_2]),
            edges: None,
            faces: (!edge.get_faces().is_empty()).then(|| edge.get_faces().clone()),
        }
    }

    pub fn from_face<T>(face: &mesh::Face<T>) -> Self {
        Self {
            nodes: None,
            verts: (!face.get_verts().is_empty()).then(|| face.get_verts().clone()),
            edges: None,
            faces: None,
        }
    }
}

impl DrawUI for MeshElementReferences {
    type ExtraData = ();

    fn draw_ui(&self, _extra_data: &Self::ExtraData, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Node");
            ui.label(self.nodes.as_ref().map_or_else(
                || "None".to_string(),
                |nodes| {
                    format!(
                        "{:?}",
                        nodes
                            .iter()
                            .map(|node_index| node_index.0.into_raw_parts().0)
                            .collect::<Vec<_>>()
                    )
                },
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Vert");
            ui.label(self.verts.as_ref().map_or_else(
                || "None".to_string(),
                |verts| {
                    format!(
                        "{:?}",
                        verts
                            .iter()
                            .map(|vert_index| vert_index.0.into_raw_parts().0)
                            .collect::<Vec<_>>()
                    )
                },
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Edge");
            ui.label(self.edges.as_ref().map_or_else(
                || "None".to_string(),
                |edges| {
                    format!(
                        "{:?}",
                        edges
                            .iter()
                            .map(|edge_index| edge_index.0.into_raw_parts().0)
                            .collect::<Vec<_>>()
                    )
                },
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Face");
            ui.label(self.faces.as_ref().map_or_else(
                || "None".to_string(),
                |faces| {
                    format!(
                        "{:?}",
                        faces
                            .iter()
                            .map(|face_index| face_index.0.into_raw_parts().0)
                            .collect::<Vec<_>>()
                    )
                },
            ));
        });
    }

    fn draw_ui_edit(&mut self, _extra_data: &Self::ExtraData, _ui: &mut Ui) {
        unreachable!()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshToBlenderMeshIndexMap {
    node_pos_index_map: HashMap<io_structs::NodeIndex, usize>,
    vert_pos_index_map: HashMap<io_structs::VertIndex, usize>,
    edge_pos_index_map: HashMap<io_structs::EdgeIndex, usize>,
    face_pos_index_map: HashMap<io_structs::FaceIndex, usize>,

    node_pos_index_inverted_map: HashMap<usize, io_structs::NodeIndex>,
    vert_pos_index_inverted_map: HashMap<usize, io_structs::VertIndex>,
    edge_pos_index_inverted_map: HashMap<usize, io_structs::EdgeIndex>,
    face_pos_index_inverted_map: HashMap<usize, io_structs::FaceIndex>,
}

impl MeshToBlenderMeshIndexMap {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_pos_index_map: HashMap<io_structs::NodeIndex, usize>,
        vert_pos_index_map: HashMap<io_structs::VertIndex, usize>,
        edge_pos_index_map: HashMap<io_structs::EdgeIndex, usize>,
        face_pos_index_map: HashMap<io_structs::FaceIndex, usize>,
        node_pos_index_inverted_map: HashMap<usize, io_structs::NodeIndex>,
        vert_pos_index_inverted_map: HashMap<usize, io_structs::VertIndex>,
        edge_pos_index_inverted_map: HashMap<usize, io_structs::EdgeIndex>,
        face_pos_index_inverted_map: HashMap<usize, io_structs::FaceIndex>,
    ) -> Self {
        Self {
            node_pos_index_map,
            vert_pos_index_map,
            edge_pos_index_map,
            face_pos_index_map,

            node_pos_index_inverted_map,
            vert_pos_index_inverted_map,
            edge_pos_index_inverted_map,
            face_pos_index_inverted_map,
        }
    }

    pub fn from_blender_mesh<END, EVD, EED, EFD>(
        io_mesh: &io_structs::Mesh<END, EVD, EED, EFD>,
    ) -> Self {
        let node_pos_index_inverted_map = io_mesh
            .node_pos_index_map
            .iter()
            .map(|(index, pos)| (*pos, *index))
            .collect();
        let vert_pos_index_inverted_map = io_mesh
            .vert_pos_index_map
            .iter()
            .map(|(index, pos)| (*pos, *index))
            .collect();
        let edge_pos_index_inverted_map = io_mesh
            .edge_pos_index_map
            .iter()
            .map(|(index, pos)| (*pos, *index))
            .collect();
        let face_pos_index_inverted_map = io_mesh
            .face_pos_index_map
            .iter()
            .map(|(index, pos)| (*pos, *index))
            .collect();

        Self::new(
            io_mesh.node_pos_index_map.clone(),
            io_mesh.vert_pos_index_map.clone(),
            io_mesh.edge_pos_index_map.clone(),
            io_mesh.face_pos_index_map.clone(),
            node_pos_index_inverted_map,
            vert_pos_index_inverted_map,
            edge_pos_index_inverted_map,
            face_pos_index_inverted_map,
        )
    }

    pub fn get_node_pos_index_map(&self) -> &HashMap<io_structs::NodeIndex, usize> {
        &self.node_pos_index_map
    }

    pub fn get_vert_pos_index_map(&self) -> &HashMap<io_structs::VertIndex, usize> {
        &self.vert_pos_index_map
    }

    pub fn get_edge_pos_index_map(&self) -> &HashMap<io_structs::EdgeIndex, usize> {
        &self.edge_pos_index_map
    }

    pub fn get_face_pos_index_map(&self) -> &HashMap<io_structs::FaceIndex, usize> {
        &self.face_pos_index_map
    }

    pub fn get_node_pos_index_inverted_map(&self) -> &HashMap<usize, io_structs::NodeIndex> {
        &self.node_pos_index_inverted_map
    }

    pub fn get_vert_pos_index_inverted_map(&self) -> &HashMap<usize, io_structs::VertIndex> {
        &self.vert_pos_index_inverted_map
    }

    pub fn get_edge_pos_index_inverted_map(&self) -> &HashMap<usize, io_structs::EdgeIndex> {
        &self.edge_pos_index_inverted_map
    }

    pub fn get_face_pos_index_inverted_map(&self) -> &HashMap<usize, io_structs::FaceIndex> {
        &self.face_pos_index_inverted_map
    }
}

pub trait MeshExtension<'de, END, EVD, EED, EFD> {
    type Error;

    fn read_file<P: AsRef<Path>>(path: P) -> Result<(Self, MeshToBlenderMeshIndexMap), Self::Error>
    where
        Self: Sized;
    fn read_msgpack<P: AsRef<Path>>(
        path: P,
    ) -> Result<(Self, MeshToBlenderMeshIndexMap), Self::Error>
    where
        Self: Sized;

    fn draw_uv(&self, draw_data: &mut MeshUVDrawData);

    fn get_node_unknown_gen(&self, index: usize) -> Result<&mesh::Node<END>, MeshExtensionError>;
    fn get_vert_unknown_gen(&self, index: usize) -> Result<&mesh::Vert<EVD>, MeshExtensionError>;
    fn get_edge_unknown_gen(&self, index: usize) -> Result<&mesh::Edge<EED>, MeshExtensionError>;
    fn get_face_unknown_gen(&self, index: usize) -> Result<&mesh::Face<EFD>, MeshExtensionError>;

    /// Visualizes the configuration given
    fn visualize_config(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError>;

    /// Gives the set of references for the given element and element_index
    ///
    /// # Example
    ///
    /// Node stores, references to the verts alone, so only that will be populated
    fn get_references_in_element(
        &self,
        element: config::Element,
        element_index: usize,
    ) -> Result<MeshElementReferences, MeshExtensionError>;

    /// Computes the aspect ratio based on various metrics and returns all of them
    fn compute_aspect_ratio_uv(&self, face: &mesh::Face<EFD>) -> (f64, f64, f64);
}

impl<
        'de,
        END: serde::Deserialize<'de> + std::fmt::Debug + Clone,
        EVD: serde::Deserialize<'de> + std::fmt::Debug + Clone,
        EED: serde::Deserialize<'de> + std::fmt::Debug + Clone,
        EFD: serde::Deserialize<'de> + std::fmt::Debug + Clone,
    > MeshExtension<'de, END, EVD, EED, EFD> for mesh::Mesh<END, EVD, EED, EFD>
{
    type Error = MeshExtensionError;

    fn read_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<(Self, MeshToBlenderMeshIndexMap), Self::Error> {
        match path.as_ref().extension() {
            Some(extension) => match extension.to_str().unwrap() {
                "mesh" => Self::read_msgpack(path),
                _ => Err(MeshExtensionError::FileExtensionUnknown),
            },
            None => Err(MeshExtensionError::NoFileExtension),
        }
    }

    fn read_msgpack<P: AsRef<Path>>(
        path: P,
    ) -> Result<(Self, MeshToBlenderMeshIndexMap), Self::Error> {
        let file = std::fs::read(path).unwrap();

        let mut cursor = std::io::Cursor::new(&file);

        let mut mesh_type = String::new();
        cursor.read_line(&mut mesh_type).unwrap();

        let mut de = Deserializer::new(cursor);

        let meshio: io_structs::Mesh<END, EVD, EED, EFD> =
            serde_path_to_error::deserialize(&mut de)?;

        let mesh_to_blender_mesh_index_map = MeshToBlenderMeshIndexMap::from_blender_mesh(&meshio);

        let mesh_res: Result<Self, ConversionError> = meshio.try_into();

        let mesh = mesh_res?;

        Ok((mesh, mesh_to_blender_mesh_index_map))
    }

    fn draw_uv(&self, draw_data: &mut MeshUVDrawData) {
        let imm = &mut draw_data.imm;
        let uv_plane_3d_model_matrix = &draw_data.uv_plane_3d_model_matrix;
        let color = &draw_data.color;

        let smooth_color_3d_shader = shader::builtins::get_smooth_color_3d_shader()
            .as_ref()
            .unwrap();

        let format = imm.get_cleared_vertex_format();
        let pos_attr = format.add_attribute(
            "in_pos\0".to_string(),
            GPUVertCompType::F32,
            3,
            GPUVertFetchMode::Float,
        );
        let color_attr = format.add_attribute(
            "in_color\0".to_string(),
            GPUVertCompType::F32,
            4,
            GPUVertFetchMode::Float,
        );

        smooth_color_3d_shader.use_shader();

        smooth_color_3d_shader.set_mat4("model\0", uv_plane_3d_model_matrix);

        if !self.get_edges().is_empty() {
            imm.begin(
                GPUPrimType::Lines,
                self.get_edges().len() * 2,
                smooth_color_3d_shader,
            );

            self.get_edges()
                .iter()
                .for_each(|(_, edge)| match edge.get_verts() {
                    Some((v1_index, v2_index)) => {
                        let v1 = self.get_vert(*v1_index).unwrap();
                        let v2 = self.get_vert(*v2_index).unwrap();

                        let uv_1_pos: glm::Vec3 =
                            glm::convert(glm::vec2_to_vec3(v1.uv.as_ref().unwrap()));
                        let uv_2_pos: glm::Vec3 =
                            glm::convert(glm::vec2_to_vec3(v2.uv.as_ref().unwrap()));

                        imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
                        imm.vertex_3f(pos_attr, uv_1_pos[0], uv_1_pos[1], uv_1_pos[2]);

                        imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
                        imm.vertex_3f(pos_attr, uv_2_pos[0], uv_2_pos[1], uv_2_pos[2]);
                    }
                    None => {}
                });

            imm.end();
        }

        if !self.get_faces().is_empty() {
            // draw the faces
            imm.begin(
                GPUPrimType::Tris,
                self.get_faces().len() * 3,
                smooth_color_3d_shader,
            );

            self.get_faces().iter().for_each(|(_, face)| {
                assert_eq!(face.get_verts().len(), 3);
                face.get_verts().iter().for_each(|vert_index| {
                    let vert = self.get_vert(*vert_index).unwrap();

                    let uv_pos: glm::Vec3 =
                        glm::convert(glm::vec2_to_vec3(vert.uv.as_ref().unwrap()));

                    imm.attr_4f(color_attr, 0.6, 0.6, 0.6, 0.03);
                    imm.vertex_3f(pos_attr, uv_pos[0], uv_pos[1], uv_pos[2]);
                });
            });

            imm.end();
        }
    }

    fn get_node_unknown_gen(&self, index: usize) -> Result<&mesh::Node<END>, MeshExtensionError> {
        Ok(self
            .get_nodes()
            .get_unknown_gen(index)
            .ok_or(MeshExtensionError::NoElementAtIndex(index))?
            .0)
    }

    fn get_vert_unknown_gen(&self, index: usize) -> Result<&mesh::Vert<EVD>, MeshExtensionError> {
        Ok(self
            .get_verts()
            .get_unknown_gen(index)
            .ok_or(MeshExtensionError::NoElementAtIndex(index))?
            .0)
    }

    fn get_edge_unknown_gen(&self, index: usize) -> Result<&mesh::Edge<EED>, MeshExtensionError> {
        Ok(self
            .get_edges()
            .get_unknown_gen(index)
            .ok_or(MeshExtensionError::NoElementAtIndex(index))?
            .0)
    }

    fn get_face_unknown_gen(&self, index: usize) -> Result<&mesh::Face<EFD>, MeshExtensionError> {
        Ok(self
            .get_faces()
            .get_unknown_gen(index)
            .ok_or(MeshExtensionError::NoElementAtIndex(index))?
            .0)
    }

    fn visualize_config(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError> {
        if config.get_draw_loose_edges() {
            let uv_plane_3d_model_matrix = config.get_uv_plane_3d_transform().get_matrix();
            let mesh_3d_model_matrix = config.get_mesh_transform().get_matrix();
            self.get_edges()
                .iter()
                .filter(|(_, edge)| edge.is_loose())
                .for_each(|(_, edge)| {
                    self.draw_fancy_edge(
                        edge,
                        &uv_plane_3d_model_matrix,
                        imm,
                        glm::convert(config.get_loose_edge_color()),
                        config.get_normal_pull_factor() * 0.8,
                    );

                    self.draw_fancy_node_edge(
                        edge,
                        &mesh_3d_model_matrix,
                        imm,
                        glm::convert(config.get_loose_edge_color()),
                        config.get_normal_pull_factor() * 0.8,
                    );
                });
        }

        if config.get_draw_loose_nodes() {
            let mesh_3d_model_matrix = config.get_mesh_transform().get_matrix();
            self.get_nodes()
                .iter()
                .filter(|(_, node)| node.is_loose())
                .for_each(|(_, node)| {
                    self.draw_fancy_node(
                        node,
                        &mesh_3d_model_matrix,
                        glm::convert(config.get_loose_node_color()),
                        imm,
                    );
                });
        }

        if config.get_draw_loose_verts() {
            let uv_plane_3d_model_matrix = config.get_uv_plane_3d_transform().get_matrix();
            self.get_verts()
                .iter()
                .filter(|(_, vert)| vert.is_loose())
                .for_each(|(_, vert)| {
                    self.draw_fancy_vert(
                        vert,
                        &uv_plane_3d_model_matrix,
                        glm::convert(config.get_loose_vert_color()),
                        imm,
                    );
                });
        }

        if config.get_draw_face_normals() && !self.get_faces().is_empty() {
            let mesh_3d_model_matrix = config.get_mesh_transform().get_matrix();

            let color: glm::Vec4 = glm::convert(config.get_face_normal_color());

            let smooth_color_3d_shader = shader::builtins::get_smooth_color_3d_shader()
                .as_ref()
                .unwrap();

            smooth_color_3d_shader.use_shader();
            smooth_color_3d_shader.set_mat4("model\0", &glm::identity());

            let format = imm.get_cleared_vertex_format();
            let pos_attr = format.add_attribute(
                "in_pos\0".to_string(),
                GPUVertCompType::F32,
                3,
                GPUVertFetchMode::Float,
            );
            let color_attr = format.add_attribute(
                "in_color\0".to_string(),
                GPUVertCompType::F32,
                4,
                GPUVertFetchMode::Float,
            );

            imm.begin_at_most(
                GPUPrimType::Lines,
                self.get_faces().len() * 2,
                smooth_color_3d_shader,
            );

            self.get_faces()
                .iter()
                .filter(|(_, face)| face.get_verts().len() >= 3)
                .for_each(|(_, face)| {
                    let normal_start_pos =
                        face.get_verts()
                            .iter()
                            .fold(glm::zero(), |acc: glm::DVec3, vert_index| {
                                let n = self
                                    .get_node(
                                        self.get_vert(*vert_index).unwrap().get_node().unwrap(),
                                    )
                                    .unwrap();
                                acc + apply_model_matrix_vec3(&n.pos, &mesh_3d_model_matrix)
                            })
                            / face.get_verts().len() as f64;

                    let normal_start_pos: glm::Vec3 = glm::convert(normal_start_pos);
                    let normal_end_pos: glm::Vec3 = normal_start_pos
                        + glm::convert::<_, glm::Vec3>(
                            apply_model_matrix_to_normal(
                                &face.normal.unwrap(),
                                &mesh_3d_model_matrix,
                            ) * config.get_face_normal_size(),
                        );

                    imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
                    imm.vertex_3f(
                        pos_attr,
                        normal_start_pos[0],
                        normal_start_pos[1],
                        normal_start_pos[2],
                    );
                    imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
                    imm.vertex_3f(
                        pos_attr,
                        normal_end_pos[0],
                        normal_end_pos[1],
                        normal_end_pos[2],
                    );
                });

            imm.end();
        }

        if config.get_draw_triangles_violating_aspect_ratio() {
            let uv_plane_3d_model_matrix = &config.get_uv_plane_3d_transform().get_matrix();
            let mesh_3d_model_matrix = &config.get_mesh_transform().get_matrix();

            self.get_faces().iter().for_each(|(_, face)| {
                let aspect_ratio = self.compute_aspect_ratio_uv(face);

                let aspect_ratio = match config.get_aspect_ratio_metric() {
                    config::AspectRatioMetric::MeasureWithInterpolationError => aspect_ratio.0,
                    config::AspectRatioMetric::RatioBetweenMinMaxDimension => aspect_ratio.1,
                    config::AspectRatioMetric::DifferentMetric => aspect_ratio.2,
                };

                if aspect_ratio < config.get_aspect_ratio_min() {
                    self.draw_fancy_face(
                        face,
                        uv_plane_3d_model_matrix,
                        mesh_3d_model_matrix,
                        imm,
                        glm::convert(config.get_face_violating_aspect_ratio_color()),
                        config.get_normal_pull_factor(),
                    );
                }
            });
        }

        match config.get_element() {
            crate::config::Element::Node => {
                // TODO(ish): handle showing which verts couldn't be
                // visualized
                self.visualize_node(config, imm)?;
            }
            crate::config::Element::Vert => {
                self.visualize_vert(config, imm)?;
            }
            crate::config::Element::Edge => {
                self.visualize_edge(config, imm)?;
            }
            crate::config::Element::Face => {
                self.visualize_face(config, imm)?;
            }
        }

        Ok(())
    }

    fn get_references_in_element(
        &self,
        element: config::Element,
        element_index: usize,
    ) -> Result<MeshElementReferences, MeshExtensionError> {
        match element {
            crate::config::Element::Node => {
                let node = self.get_node_unknown_gen(element_index)?;
                Ok(MeshElementReferences::from_node(node))
            }
            crate::config::Element::Vert => {
                let vert = self.get_vert_unknown_gen(element_index)?;
                Ok(MeshElementReferences::from_vert(vert))
            }
            crate::config::Element::Edge => {
                let edge = self.get_edge_unknown_gen(element_index)?;
                Ok(MeshElementReferences::from_edge(edge))
            }
            crate::config::Element::Face => {
                let face = self.get_face_unknown_gen(element_index)?;
                Ok(MeshElementReferences::from_face(face))
            }
        }
    }

    fn compute_aspect_ratio_uv(&self, face: &mesh::Face<EFD>) -> (f64, f64, f64) {
        let verts = &face.get_verts();

        let vert_1_index = verts[0];
        let vert_1 = self.get_vert(vert_1_index).unwrap();

        let vert_2_index = verts[1];
        let vert_2 = self.get_vert(vert_2_index).unwrap();

        let vert_3_index = verts[2];
        let vert_3 = self.get_vert(vert_3_index).unwrap();

        let p1 = vert_1.uv.unwrap();
        let p2 = vert_2.uv.unwrap();
        let p3 = vert_3.uv.unwrap();

        let l1 = (p2 - p1).norm();
        let l2 = (p3 - p2).norm();
        let l3 = (p1 - p3).norm();
        let area = 0.5 * glm::cross2d(&(p2 - p1), &(p3 - p1)).abs();
        let perimeter = l1 + l2 + l3;
        let l_max = l1.max(l2).max(l3);

        // the measure associated with
        // interpolation error
        let metric_1 = || (4.0 * 3.0_f64.sqrt() * area) / (l_max * perimeter);

        // aspect ratio or ratio between
        // minumum and maxium dimension of
        // triangle
        let metric_2 = || (4.0 * area) / (3.0_f64.sqrt() * l_max.powi(2));

        // different metric
        let metric_3 = || 12.0 * 3.0_f64.sqrt() * area / perimeter.powi(2);

        (metric_1(), metric_2(), metric_3())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshExtensionError {
    NoElementAtIndex(usize),
    DeserializationError(String),
    FileExtensionUnknown,
    NoFileExtension,
    NoMesh,
    NoExtraData,
    ConversionFailed(ConversionError),
    UnknownMeshType,
}

impl std::error::Error for MeshExtensionError {}

impl Display for MeshExtensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshExtensionError::NoElementAtIndex(index) => {
                write!(f, "No element at index: {}", index)
            }
            MeshExtensionError::DeserializationError(error) => {
                write!(f, "Deserialization error: {}", error)
            }
            MeshExtensionError::FileExtensionUnknown => {
                write!(f, "File extension unknown")
            }
            MeshExtensionError::NoFileExtension => {
                write!(f, "No file extension so probably not a file")
            }
            MeshExtensionError::NoMesh => {
                write!(f, "No mesh available")
            }
            MeshExtensionError::NoExtraData => {
                write!(f, "No extra data for the element")
            }
            MeshExtensionError::ConversionFailed(error) => {
                write!(f, "Conversion failed: {}", error)
            }
            MeshExtensionError::UnknownMeshType => {
                write!(f, "Unknown mesh type")
            }
        }
    }
}

impl From<serde_path_to_error::Error<rmps::decode::Error>> for MeshExtensionError {
    fn from(error: serde_path_to_error::Error<rmps::decode::Error>) -> Self {
        Self::DeserializationError(error.to_string())
    }
}

impl From<ConversionError> for MeshExtensionError {
    fn from(error: ConversionError) -> Self {
        Self::ConversionFailed(error)
    }
}

trait MeshExtensionPrivate<END, EVD, EED, EFD> {
    /// Tries to visualize all the links stored in the node
    /// that refer to it.
    ///
    /// It returns back all the verts that cannot be visualized.
    fn visualize_node(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Result<Vec<mesh::VertIndex>, MeshExtensionError>;

    /// Tries to visualize all the links stored in the vert
    /// that refer to it.
    ///
    /// It returns back all the edges and node that cannot be visualized.
    /// TODO(ish): the returning part
    fn visualize_vert(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError>;

    /// Tries to visualize all the links stored in the edge
    /// that refer to it.
    ///
    /// It returns back all the verts and faces that cannot be visualized.
    /// TODO(ish): the returning part
    fn visualize_edge(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError>;

    /// Tries to visualize all the links stored in the face
    /// that refer to it.
    ///
    /// It returns back all the references that cannot be visualized.
    /// TODO(ish): the returning part
    fn visualize_face(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError>;
}

impl<END, EVD, EED, EFD> MeshExtensionPrivate<END, EVD, EED, EFD>
    for mesh::Mesh<END, EVD, EED, EFD>
{
    fn visualize_node(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Result<Vec<mesh::VertIndex>, MeshExtensionError> {
        let uv_plane_3d_model_matrix = &config.get_uv_plane_3d_transform().get_matrix();
        let mesh_model_matrix = &config.get_mesh_transform().get_matrix();

        let mut no_uv_verts = Vec::new();
        let node = self
            .get_nodes()
            .get_unknown_gen(config.get_element_index())
            .ok_or_else(|| MeshExtensionError::NoElementAtIndex(config.get_element_index()))?
            .0;
        node.get_verts().iter().for_each(|vert_index| {
            let vert = self.get_vert(*vert_index).unwrap();
            match vert.uv {
                Some(_) => {
                    self.draw_fancy_node_vert_connect(
                        node,
                        vert,
                        uv_plane_3d_model_matrix,
                        mesh_model_matrix,
                        imm,
                        glm::convert(config.get_node_color()),
                        glm::convert(config.get_vert_color()),
                        glm::convert(config.get_node_vert_connect_color()),
                        config.get_normal_pull_factor(),
                    );
                }
                None => no_uv_verts.push(*vert_index),
            }
        });
        Ok(no_uv_verts)
    }

    fn visualize_vert(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError> {
        let uv_plane_3d_model_matrix = &config.get_uv_plane_3d_transform().get_matrix();
        let mesh_model_matrix = &config.get_mesh_transform().get_matrix();

        let vert = self
            .get_verts()
            .get_unknown_gen(config.get_element_index())
            .ok_or_else(|| MeshExtensionError::NoElementAtIndex(config.get_element_index()))?
            .0;

        vert.get_edges().iter().for_each(|edge_index| {
            let edge = self.get_edge(*edge_index).unwrap();

            self.draw_fancy_edge(
                edge,
                uv_plane_3d_model_matrix,
                imm,
                glm::convert(config.get_edge_color()),
                config.get_normal_pull_factor(),
            );
        });

        let node = self.get_node(vert.get_node().unwrap()).unwrap();
        self.draw_fancy_node_vert_connect(
            node,
            vert,
            uv_plane_3d_model_matrix,
            mesh_model_matrix,
            imm,
            glm::convert(config.get_node_color()),
            glm::convert(config.get_vert_color()),
            glm::convert(config.get_node_vert_connect_color()),
            config.get_normal_pull_factor(),
        );

        Ok(())
    }

    fn visualize_edge(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError> {
        let uv_plane_3d_model_matrix = &config.get_uv_plane_3d_transform().get_matrix();
        let mesh_model_matrix = &config.get_mesh_transform().get_matrix();

        let edge = self
            .get_edges()
            .get_unknown_gen(config.get_element_index())
            .ok_or_else(|| MeshExtensionError::NoElementAtIndex(config.get_element_index()))?
            .0;

        self.draw_fancy_edge(
            edge,
            uv_plane_3d_model_matrix,
            imm,
            glm::convert(config.get_edge_color()),
            config.get_normal_pull_factor(),
        );

        self.draw_fancy_node_edge(
            edge,
            mesh_model_matrix,
            imm,
            glm::convert(config.get_edge_color()),
            config.get_normal_pull_factor(),
        );

        edge.get_faces()
            .iter()
            .enumerate()
            .for_each(|(i, face_index)| {
                let face_colors = &config.get_face_color();
                let face_color = glm::convert(face_colors.0.lerp(
                    &face_colors.1,
                    i as f64 / (edge.get_faces().len() - 1).max(1) as f64,
                ));
                let face = self.get_face(*face_index).unwrap();
                self.draw_fancy_face(
                    face,
                    uv_plane_3d_model_matrix,
                    mesh_model_matrix,
                    imm,
                    face_color,
                    config.get_normal_pull_factor(),
                );
            });

        Ok(())
    }

    fn visualize_face(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Result<(), MeshExtensionError> {
        let uv_plane_3d_model_matrix = &config.get_uv_plane_3d_transform().get_matrix();
        let mesh_model_matrix = &config.get_mesh_transform().get_matrix();

        let face = self
            .get_faces()
            .get_unknown_gen(config.get_element_index())
            .ok_or_else(|| MeshExtensionError::NoElementAtIndex(config.get_element_index()))?
            .0;

        self.draw_fancy_face(
            face,
            uv_plane_3d_model_matrix,
            mesh_model_matrix,
            imm,
            glm::convert(config.get_face_color().0),
            config.get_normal_pull_factor(),
        );

        Ok(())
    }
}

fn draw_sphere_at(pos: &glm::DVec3, color: glm::Vec4, imm: &mut GPUImmediate) {
    let smooth_color_3d_shader = get_smooth_color_3d_shader().as_ref().unwrap();
    smooth_color_3d_shader.use_shader();
    smooth_color_3d_shader.set_mat4(
        "model\0",
        &glm::convert(glm::scale(
            &glm::translate(&glm::identity(), pos),
            &glm::vec3(0.02, 0.02, 0.02),
        )),
    );

    let ico_sphere = get_ico_sphere_subd_00();

    ico_sphere
        .draw(&mut MeshDrawData::new(
            imm,
            mesh::MeshUseShader::SmoothColor3D,
            Some(color),
        ))
        .unwrap();
}

fn draw_lines(positions: &[glm::DVec3], color: glm::Vec4, imm: &mut GPUImmediate) {
    assert_ne!(positions.len(), 0);
    assert!(positions.len() % 2 == 0);
    let smooth_color_3d_shader = shader::builtins::get_smooth_color_3d_shader()
        .as_ref()
        .unwrap();

    smooth_color_3d_shader.use_shader();
    smooth_color_3d_shader.set_mat4("model\0", &glm::identity());

    let format = imm.get_cleared_vertex_format();
    let pos_attr = format.add_attribute(
        "in_pos\0".to_string(),
        GPUVertCompType::F32,
        3,
        GPUVertFetchMode::Float,
    );
    let color_attr = format.add_attribute(
        "in_color\0".to_string(),
        GPUVertCompType::F32,
        4,
        GPUVertFetchMode::Float,
    );

    imm.begin_at_most(
        GPUPrimType::Lines,
        positions.len() * 2,
        smooth_color_3d_shader,
    );

    positions.iter().for_each(|pos| {
        let pos: glm::Vec3 = glm::convert(*pos);
        imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
        imm.vertex_3f(pos_attr, pos[0], pos[1], pos[2]);
    });

    imm.end();
}

/// Draw the element only in a fancy way
trait MeshDrawFancy<END, EVD, EED, EFD> {
    fn draw_fancy_node(
        &self,
        node: &mesh::Node<END>,
        mesh_model_matrix: &glm::DMat4,
        node_color: glm::Vec4,
        imm: &mut GPUImmediate,
    );

    fn draw_fancy_vert(
        &self,
        vert: &mesh::Vert<EVD>,
        uv_plane_3d_model_matrix: &glm::DMat4,
        vert_color: glm::Vec4,
        imm: &mut GPUImmediate,
    );

    #[allow(clippy::too_many_arguments)]
    fn draw_fancy_node_vert_connect(
        &self,
        node: &mesh::Node<END>,
        vert: &mesh::Vert<EVD>,
        uv_plane_3d_model_matrix: &glm::DMat4,
        mesh_model_matrix: &glm::DMat4,
        imm: &mut GPUImmediate,
        node_color: glm::Vec4,
        vert_color: glm::Vec4,
        node_vert_connect_color: glm::Vec4,
        normal_pull_factor: f64,
    );

    fn draw_fancy_edge(
        &self,
        edge: &mesh::Edge<EED>,
        uv_plane_3d_model_matrix: &glm::DMat4,
        imm: &mut GPUImmediate,
        edge_color: glm::Vec4,
        normal_pull_factor: f64,
    );

    fn draw_fancy_node_edge(
        &self,
        edge: &mesh::Edge<EED>,
        mesh_model_matrix: &glm::DMat4,
        imm: &mut GPUImmediate,
        edge_color: glm::Vec4,
        normal_pull_factor: f64,
    );

    fn draw_fancy_face(
        &self,
        face: &mesh::Face<EFD>,
        uv_plane_3d_model_matrix: &glm::DMat4,
        mesh_model_matrix: &glm::DMat4,
        imm: &mut GPUImmediate,
        face_color: glm::Vec4,
        normal_pull_factor: f64,
    );
}

impl<END, EVD, EED, EFD> MeshDrawFancy<END, EVD, EED, EFD> for mesh::Mesh<END, EVD, EED, EFD> {
    fn draw_fancy_node(
        &self,
        node: &mesh::Node<END>,
        mesh_model_matrix: &glm::DMat4,
        node_color: glm::Vec4,
        imm: &mut GPUImmediate,
    ) {
        let node_pos_applied = apply_model_matrix_vec3(&node.pos, mesh_model_matrix);
        draw_sphere_at(&node_pos_applied, node_color, imm);
    }

    fn draw_fancy_vert(
        &self,
        vert: &mesh::Vert<EVD>,
        uv_plane_3d_model_matrix: &glm::DMat4,
        vert_color: glm::Vec4,
        imm: &mut GPUImmediate,
    ) {
        let uv = vert.uv.as_ref().unwrap();
        let uv_pos = apply_model_matrix_vec2(uv, uv_plane_3d_model_matrix);

        draw_sphere_at(&uv_pos, vert_color, imm);
    }

    fn draw_fancy_node_vert_connect(
        &self,
        node: &mesh::Node<END>,
        vert: &mesh::Vert<EVD>,
        uv_plane_3d_model_matrix: &glm::DMat4,
        mesh_model_matrix: &glm::DMat4,
        imm: &mut GPUImmediate,
        node_color: glm::Vec4,
        vert_color: glm::Vec4,
        node_vert_connect_color: glm::Vec4,
        normal_pull_factor: f64,
    ) {
        self.draw_fancy_node(node, mesh_model_matrix, node_color, imm);
        self.draw_fancy_vert(vert, uv_plane_3d_model_matrix, vert_color, imm);

        let uv = vert.uv.as_ref().unwrap();
        let uv_pos = apply_model_matrix_vec2(uv, uv_plane_3d_model_matrix);
        let initial_uv_plane_normal = glm::vec3(0.0, 0.0, 1.0);
        let uv_plane_normal_applied =
            apply_model_matrix_to_normal(&initial_uv_plane_normal, uv_plane_3d_model_matrix);

        let node_pos_applied = apply_model_matrix_vec3(&node.pos, mesh_model_matrix);
        let node_normal_applied =
            apply_model_matrix_to_normal(&node.normal.unwrap(), mesh_model_matrix);

        let curve = CubicBezierCurve::new(
            node_pos_applied,
            node_pos_applied + node_normal_applied * normal_pull_factor,
            uv_pos + uv_plane_normal_applied * normal_pull_factor,
            uv_pos,
            20,
        );

        let smooth_color_3d_shader = get_smooth_color_3d_shader().as_ref().unwrap();
        smooth_color_3d_shader.use_shader();
        smooth_color_3d_shader.set_mat4("model\0", &glm::identity());

        curve
            .draw(&mut CubicBezierCurveDrawData::new(
                imm,
                node_vert_connect_color,
            ))
            .unwrap();
    }

    fn draw_fancy_edge(
        &self,
        edge: &mesh::Edge<EED>,
        uv_plane_3d_model_matrix: &glm::DMat4,
        imm: &mut GPUImmediate,
        edge_color: glm::Vec4,
        normal_pull_factor: f64,
    ) {
        let v1_index = edge.get_verts().unwrap().0;
        let v2_index = edge.get_verts().unwrap().1;

        let v1 = self.get_vert(v1_index).unwrap();
        let v2 = self.get_vert(v2_index).unwrap();

        let v1_uv = v1.uv.unwrap();
        let v2_uv = v2.uv.unwrap();

        let v1_uv_applied = apply_model_matrix_vec2(&v1_uv, uv_plane_3d_model_matrix);
        let v2_uv_applied = apply_model_matrix_vec2(&v2_uv, uv_plane_3d_model_matrix);

        let initial_normal = glm::vec3(0.0, 0.0, 1.0);
        let normal_applied =
            apply_model_matrix_to_normal(&initial_normal, uv_plane_3d_model_matrix);

        let smooth_color_3d_shader = get_smooth_color_3d_shader().as_ref().unwrap();
        smooth_color_3d_shader.use_shader();
        smooth_color_3d_shader.set_mat4("model\0", &glm::identity());

        let curve = CubicBezierCurve::new(
            v1_uv_applied,
            v1_uv_applied + normal_applied * normal_pull_factor,
            v2_uv_applied + normal_applied * normal_pull_factor,
            v2_uv_applied,
            20,
        );

        curve
            .draw(&mut CubicBezierCurveDrawData::new(imm, edge_color))
            .unwrap();
    }

    fn draw_fancy_node_edge(
        &self,
        edge: &mesh::Edge<EED>,
        mesh_model_matrix: &glm::DMat4,
        imm: &mut GPUImmediate,
        edge_color: glm::Vec4,
        normal_pull_factor: f64,
    ) {
        let v1_index = edge.get_verts().unwrap().0;
        let v2_index = edge.get_verts().unwrap().1;

        let v1 = self.get_vert(v1_index).unwrap();
        let v2 = self.get_vert(v2_index).unwrap();

        let n1 = self.get_node(v1.get_node().unwrap()).unwrap();
        let n2 = self.get_node(v2.get_node().unwrap()).unwrap();

        let n1_pos_applied = apply_model_matrix_vec3(&n1.pos, mesh_model_matrix);
        let n2_pos_applied = apply_model_matrix_vec3(&n2.pos, mesh_model_matrix);

        let n1_normal_applied =
            apply_model_matrix_to_normal(&n1.normal.unwrap(), mesh_model_matrix);
        let n2_normal_applied =
            apply_model_matrix_to_normal(&n2.normal.unwrap(), mesh_model_matrix);

        let smooth_color_3d_shader = get_smooth_color_3d_shader().as_ref().unwrap();
        smooth_color_3d_shader.use_shader();
        smooth_color_3d_shader.set_mat4("model\0", &glm::identity());

        let curve = CubicBezierCurve::new(
            n1_pos_applied,
            n1_pos_applied + n1_normal_applied * normal_pull_factor,
            n2_pos_applied + n2_normal_applied * normal_pull_factor,
            n2_pos_applied,
            20,
        );

        curve
            .draw(&mut CubicBezierCurveDrawData::new(imm, edge_color))
            .unwrap();
    }

    fn draw_fancy_face(
        &self,
        face: &mesh::Face<EFD>,
        uv_plane_3d_model_matrix: &glm::DMat4,
        mesh_model_matrix: &glm::DMat4,
        imm: &mut GPUImmediate,
        face_color: glm::Vec4,
        normal_pull_factor: f64,
    ) {
        // Currently only support triangles
        assert_eq!(face.get_verts().len(), 3);

        let v1_index = face.get_verts()[0];
        let v2_index = face.get_verts()[1];
        let v3_index = face.get_verts()[2];

        let v1 = self.get_vert(v1_index).unwrap();
        let v2 = self.get_vert(v2_index).unwrap();
        let v3 = self.get_vert(v3_index).unwrap();

        let v1_uv = v1.uv.unwrap();
        let v2_uv = v2.uv.unwrap();
        let v3_uv = v3.uv.unwrap();

        let v1_uv_applied = apply_model_matrix_vec2(&v1_uv, uv_plane_3d_model_matrix);
        let v2_uv_applied = apply_model_matrix_vec2(&v2_uv, uv_plane_3d_model_matrix);
        let v3_uv_applied = apply_model_matrix_vec2(&v3_uv, uv_plane_3d_model_matrix);

        let triangle_center = (v1_uv_applied + v2_uv_applied + v3_uv_applied) / 3.0;

        let initial_uv_plane_normal = glm::vec3(0.0, 0.0, 1.0);
        let uv_plane_normal_applied =
            apply_model_matrix_to_normal(&initial_uv_plane_normal, uv_plane_3d_model_matrix);

        let point_away = triangle_center - uv_plane_normal_applied;

        let v1_norm = (v1_uv_applied - point_away).normalize() * 2.5 * normal_pull_factor;
        let v2_norm = (v2_uv_applied - point_away).normalize() * 2.5 * normal_pull_factor;
        let v3_norm = (v3_uv_applied - point_away).normalize() * 2.5 * normal_pull_factor;

        let pn_triangle = PointNormalTriangle::new(
            v1_uv_applied,
            v2_uv_applied,
            v3_uv_applied,
            v1_norm,
            v2_norm,
            v3_norm,
            10,
        );

        let smooth_color_3d_shader = get_smooth_color_3d_shader().as_ref().unwrap();
        smooth_color_3d_shader.use_shader();
        smooth_color_3d_shader.set_mat4("model\0", &glm::identity());

        pn_triangle
            .draw(&mut PointNormalTriangleDrawData::new(
                imm,
                face_color,
                false,
                0.2,
                glm::vec4(1.0, 0.0, 0.0, 1.0),
            ))
            .unwrap();

        // draw the face on the mesh
        {
            let poses_normals: Vec<_> = self
                .get_nodes_of_face(face)
                .iter()
                .map(|op_node_index| {
                    let node_index = op_node_index.unwrap();
                    let node = self.get_node(node_index).unwrap();

                    let node_pos_applied = apply_model_matrix_vec3(&node.pos, mesh_model_matrix);
                    let node_normal_applied =
                        apply_model_matrix_to_normal(&node.normal.unwrap(), mesh_model_matrix);

                    (node_pos_applied, node_normal_applied)
                })
                .collect();

            assert_eq!(poses_normals.len(), 3);

            let (center_tot, avg_normal_tot): (glm::DVec3, glm::DVec3) = poses_normals.iter().fold(
                (glm::zero(), glm::zero()),
                |acc: (glm::DVec3, glm::DVec3), (pos, normal)| (acc.0 + pos, acc.1 + normal),
            );
            let center = center_tot / poses_normals.len() as f64;
            let avg_normal = avg_normal_tot / poses_normals.len() as f64;

            let point_away = center - avg_normal;

            let pn_triangle = PointNormalTriangle::new(
                poses_normals[0].0,
                poses_normals[1].0,
                poses_normals[2].0,
                (poses_normals[0].0 - point_away).normalize() * 2.5 * normal_pull_factor,
                (poses_normals[1].0 - point_away).normalize() * 2.5 * normal_pull_factor,
                (poses_normals[2].0 - point_away).normalize() * 2.5 * normal_pull_factor,
                10,
            );

            pn_triangle
                .draw(&mut PointNormalTriangleDrawData::new(
                    imm,
                    face_color,
                    false,
                    0.2,
                    glm::vec4(1.0, 0.0, 0.0, 1.0),
                ))
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_read_file_msgpack() {
        let mesh = mesh::simple::Mesh::read_file("/tmp/test.msgpack").unwrap();
        println!("mesh: {:?}", mesh);
    }
}
