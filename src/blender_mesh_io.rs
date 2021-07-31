use quick_renderer::mesh::builtins::get_ico_sphere_subd_00;
use quick_renderer::mesh::MeshDrawData;
use quick_renderer::shader::builtins::get_smooth_color_3d_shader;
use rmps::Deserializer;
use serde::Deserialize;

use std::path::Path;

use quick_renderer::drawable::Drawable;
use quick_renderer::gpu_immediate::{GPUImmediate, GPUPrimType, GPUVertCompType, GPUVertFetchMode};
use quick_renderer::{glm, mesh, shader};

use crate::curve::{PointNormalTriangle, PointNormalTriangleDrawData};
use crate::math;
use crate::{
    config::Config,
    curve::{CubicBezierCurve, CubicBezierCurveDrawData},
};

mod io_structs {
    use std::collections::HashMap;

    use generational_arena::Arena;
    use quick_renderer::{glm, mesh};
    use serde::{Deserialize, Serialize};

    use crate::util;

    #[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
    pub(super) struct Float3 {
        x: f32,
        y: f32,
        z: f32,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
    pub(super) struct Float2 {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub(super) struct Index {
        index: usize,
        generation: usize,
    }

    pub(super) type NodeIndex = Index;
    pub(super) type VertIndex = Index;
    pub(super) type EdgeIndex = Index;
    pub(super) type FaceIndex = Index;

    pub(super) type IncidentVerts = Vec<VertIndex>;
    pub(super) type IncidentEdges = Vec<EdgeIndex>;
    pub(super) type IncidentFaces = Vec<FaceIndex>;
    pub(super) type AdjacentVerts = IncidentVerts;
    pub(super) type EdgeVerts = (VertIndex, VertIndex);

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Node<T> {
        self_index: NodeIndex,
        verts: IncidentVerts,

        pos: Float3,
        normal: Float3,
        extra_data: Option<T>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Vert<T> {
        self_index: VertIndex,
        edges: IncidentEdges,
        node: Option<NodeIndex>,

        uv: Float2,
        extra_data: Option<T>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Edge<T> {
        self_index: EdgeIndex,
        faces: IncidentFaces,
        verts: Option<EdgeVerts>,

        extra_data: Option<T>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Face<T> {
        self_index: FaceIndex,
        verts: AdjacentVerts,

        normal: Float3,
        extra_data: Option<T>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Mesh<END, EVD, EED, EFD> {
        nodes: Vec<Node<END>>,
        verts: Vec<Vert<EVD>>,
        edges: Vec<Edge<EED>>,
        faces: Vec<Face<EFD>>,

        node_pos_index_map: HashMap<NodeIndex, usize>,
        vert_pos_index_map: HashMap<VertIndex, usize>,
        edge_pos_index_map: HashMap<EdgeIndex, usize>,
        face_pos_index_map: HashMap<FaceIndex, usize>,
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

    impl<END, EVD, EED, EFD> From<Mesh<END, EVD, EED, EFD>> for mesh::Mesh<END, EVD, EED, EFD> {
        fn from(mut io_mesh: Mesh<END, EVD, EED, EFD>) -> Self {
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
                .for_each(|(node, io_node)| {
                    let node_verts;
                    unsafe {
                        node_verts = node.1.get_verts_mut();
                    }
                    io_node.verts.iter().for_each(|io_node_vert_index| {
                        let vert_i = io_mesh.vert_pos_index_map.get(io_node_vert_index).unwrap();
                        node_verts.push(mesh::VertIndex(verts.get_unknown_gen(*vert_i).unwrap().1));
                    });
                });

            verts
                .iter_mut()
                .zip(io_mesh.verts.iter())
                .for_each(|(vert, io_vert)| {
                    let vert_edges;
                    unsafe {
                        vert_edges = vert.1.get_edges_mut();
                    }
                    io_vert.edges.iter().for_each(|io_vert_edge_index| {
                        let edge_i = io_mesh.edge_pos_index_map.get(io_vert_edge_index).unwrap();
                        vert_edges.push(mesh::EdgeIndex(edges.get_unknown_gen(*edge_i).unwrap().1));
                    });

                    let vert_node;
                    unsafe {
                        vert_node = vert.1.get_node_mut();
                    }

                    *vert_node = io_vert.node.as_ref().map(|io_vert_node_index| {
                        let node_i = io_mesh.node_pos_index_map.get(io_vert_node_index).unwrap();
                        mesh::NodeIndex(nodes.get_unknown_gen(*node_i).unwrap().1)
                    });
                });

            edges
                .iter_mut()
                .zip(io_mesh.edges.iter())
                .for_each(|(edge, io_edge)| {
                    let edge_verts;
                    unsafe {
                        edge_verts = edge.1.get_verts_mut();
                    }
                    *edge_verts =
                        io_edge
                            .verts
                            .map(|(io_edge_vert_1_index, io_edge_vert_2_index)| {
                                let edge_vert_1_i = io_mesh
                                    .vert_pos_index_map
                                    .get(&io_edge_vert_1_index)
                                    .unwrap();

                                let edge_vert_2_i = io_mesh
                                    .vert_pos_index_map
                                    .get(&io_edge_vert_2_index)
                                    .unwrap();

                                (
                                    mesh::VertIndex(
                                        verts.get_unknown_gen(*edge_vert_1_i).unwrap().1,
                                    ),
                                    mesh::VertIndex(
                                        verts.get_unknown_gen(*edge_vert_2_i).unwrap().1,
                                    ),
                                )
                            });

                    let edge_faces;
                    unsafe {
                        edge_faces = edge.1.get_faces_mut();
                    }
                    io_edge.faces.iter().for_each(|io_edge_face_index| {
                        let face_i = io_mesh.face_pos_index_map.get(io_edge_face_index).unwrap();
                        edge_faces.push(mesh::FaceIndex(faces.get_unknown_gen(*face_i).unwrap().1))
                    });
                });

            faces
                .iter_mut()
                .zip(io_mesh.faces.iter())
                .for_each(|(face, io_face)| {
                    let face_verts;
                    unsafe {
                        face_verts = face.1.get_verts_mut();
                    }
                    io_face.verts.iter().for_each(|io_face_vert_index| {
                        let vert_i = io_mesh.vert_pos_index_map.get(io_face_vert_index).unwrap();
                        face_verts.push(mesh::VertIndex(verts.get_unknown_gen(*vert_i).unwrap().1));
                    });
                });

            let mut mesh = Self::from_arenas(nodes, verts, edges, faces);
            // since the mesh is from Blender, we need to convert the
            // mesh coordinates from Blender's to OpenGL
            mesh.apply_model_matrix(&util::axis_conversion_matrix(
                util::Axis::Y,
                util::Axis::Z,
                util::Axis::NegZ,
                util::Axis::Y,
            ));
            mesh
        }
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

pub trait MeshExtension<'de, END, EVD, EED, EFD> {
    type Error;

    fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, Self::Error>
    where
        Self: Sized;
    fn read_msgpack<P: AsRef<Path>>(path: P) -> Result<Self, Self::Error>
    where
        Self: Sized;

    fn draw_uv(&self, draw_data: &mut MeshUVDrawData);

    /// Visualizes the configuration given
    fn visualize_config(&self, config: &Config<END, EVD, EED, EFD>, imm: &mut GPUImmediate);
}

impl<
        'de,
        END: serde::Deserialize<'de> + std::fmt::Debug,
        EVD: serde::Deserialize<'de> + std::fmt::Debug,
        EED: serde::Deserialize<'de> + std::fmt::Debug,
        EFD: serde::Deserialize<'de> + std::fmt::Debug,
    > MeshExtension<'de, END, EVD, EED, EFD> for mesh::Mesh<END, EVD, EED, EFD>
{
    type Error = ();

    fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, ()> {
        match path.as_ref().extension() {
            Some(extension) => match extension.to_str().unwrap() {
                "msgpack" => Self::read_msgpack(path),
                _ => Err(()),
            },
            None => Err(()),
        }
    }
    fn read_msgpack<P: AsRef<Path>>(path: P) -> Result<Self, ()> {
        let file = std::fs::read(path).unwrap();

        let mut de = Deserializer::new(std::io::Cursor::new(&file));

        let meshio: io_structs::Mesh<END, EVD, EED, EFD> =
            Deserialize::deserialize(&mut de).unwrap();

        let mesh: Self = meshio.into();

        Ok(mesh)
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

    fn visualize_config(&self, config: &Config<END, EVD, EED, EFD>, imm: &mut GPUImmediate) {
        match config.get_element() {
            crate::config::Element::Node => {
                // TODO(ish): handle showing which verts couldn't be
                // visualized
                self.visualize_node(config, imm);
            }
            crate::config::Element::Vert => {
                self.visualize_vert(config, imm);
            }
            crate::config::Element::Edge => {
                self.visualize_edge(config, imm);
            }
            crate::config::Element::Face => {
                self.visualize_face(config, imm);
            }
        }
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
    ) -> Vec<mesh::VertIndex>;

    /// Tries to visualize all the links stored in the vert
    /// that refer to it.
    ///
    /// It returns back all the edges and node that cannot be visualized.
    /// TODO(ish): the returning part
    fn visualize_vert(&self, config: &Config<END, EVD, EED, EFD>, imm: &mut GPUImmediate);

    /// Tries to visualize all the links stored in the edge
    /// that refer to it.
    ///
    /// It returns back all the verts and faces that cannot be visualized.
    /// TODO(ish): the returning part
    fn visualize_edge(&self, config: &Config<END, EVD, EED, EFD>, imm: &mut GPUImmediate);

    /// Tries to visualize all the links stored in the face
    /// that refer to it.
    ///
    /// It returns back all the references that cannot be visualized.
    /// TODO(ish): the returning part
    fn visualize_face(&self, config: &Config<END, EVD, EED, EFD>, imm: &mut GPUImmediate);
}

impl<END, EVD, EED, EFD> MeshExtensionPrivate<END, EVD, EED, EFD>
    for mesh::Mesh<END, EVD, EED, EFD>
{
    fn visualize_node(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Vec<mesh::VertIndex> {
        let normal_pull_factor = 0.2;

        let uv_plane_3d_model_matrix = &config.get_uv_plane_3d_transform().get_matrix();
        let mesh_model_matrix = &config.get_mesh_transform().get_matrix();

        let mut no_uv_verts = Vec::new();
        let node = self
            .get_nodes()
            .get_unknown_gen(config.get_element_index())
            .unwrap()
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
                        normal_pull_factor,
                    );
                }
                None => no_uv_verts.push(*vert_index),
            }
        });
        no_uv_verts
    }

    fn visualize_vert(&self, config: &Config<END, EVD, EED, EFD>, imm: &mut GPUImmediate) {
        let normal_pull_factor = 0.2;

        let uv_plane_3d_model_matrix = &config.get_uv_plane_3d_transform().get_matrix();
        let mesh_model_matrix = &config.get_mesh_transform().get_matrix();

        let vert = self
            .get_verts()
            .get_unknown_gen(config.get_element_index())
            .unwrap()
            .0;

        vert.get_edges().iter().for_each(|edge_index| {
            let edge = self.get_edge(*edge_index).unwrap();

            self.draw_fancy_edge(
                edge,
                uv_plane_3d_model_matrix,
                imm,
                glm::convert(config.get_edge_color()),
                normal_pull_factor,
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
            normal_pull_factor,
        );
    }

    fn visualize_edge(&self, config: &Config<END, EVD, EED, EFD>, imm: &mut GPUImmediate) {
        let normal_pull_factor = 0.2;

        let uv_plane_3d_model_matrix = &config.get_uv_plane_3d_transform().get_matrix();
        let mesh_model_matrix = &config.get_mesh_transform().get_matrix();

        let edge = self
            .get_edges()
            .get_unknown_gen(config.get_element_index())
            .unwrap()
            .0;

        self.draw_fancy_edge(
            edge,
            uv_plane_3d_model_matrix,
            imm,
            glm::convert(config.get_edge_color()),
            normal_pull_factor,
        );

        self.draw_fancy_node_edge(
            edge,
            mesh_model_matrix,
            imm,
            glm::convert(config.get_edge_color()),
            normal_pull_factor,
        );

        edge.get_faces()
            .iter()
            .enumerate()
            .for_each(|(i, face_index)| {
                let face_colors = &config.get_face_color();
                let face_color = glm::convert(face_colors.0.lerp(
                    &face_colors.1,
                    (i as f64 / edge.get_faces().len() as f64).ceil(),
                ));
                let face = self.get_face(*face_index).unwrap();
                self.draw_fancy_face(
                    face,
                    uv_plane_3d_model_matrix,
                    mesh_model_matrix,
                    imm,
                    face_color,
                    normal_pull_factor,
                );
            });
    }

    fn visualize_face(&self, config: &Config<END, EVD, EED, EFD>, imm: &mut GPUImmediate) {
        let normal_pull_factor = 0.2;

        let uv_plane_3d_model_matrix = &config.get_uv_plane_3d_transform().get_matrix();
        let mesh_model_matrix = &config.get_mesh_transform().get_matrix();

        let face = self
            .get_faces()
            .get_unknown_gen(config.get_element_index())
            .unwrap()
            .0;

        self.draw_fancy_face(
            face,
            uv_plane_3d_model_matrix,
            mesh_model_matrix,
            imm,
            glm::convert(config.get_face_color().0),
            normal_pull_factor,
        );
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

fn apply_model_matrix_vec2(v: &glm::DVec2, model: &glm::DMat4) -> glm::DVec3 {
    glm::vec4_to_vec3(&(model * math::append_one(&glm::vec2_to_vec3(v))))
}

fn apply_model_matrix_vec3(v: &glm::DVec3, model: &glm::DMat4) -> glm::DVec3 {
    glm::vec4_to_vec3(&(model * math::append_one(v)))
}

fn apply_model_matrix_to_normal(normal: &glm::DVec3, model: &glm::DMat4) -> glm::DVec3 {
    apply_model_matrix_vec3(normal, &glm::inverse_transpose(*model))
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
