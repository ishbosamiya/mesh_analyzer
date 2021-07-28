use rmps::Deserializer;
use serde::Deserialize;

use std::path::Path;

use quick_renderer::drawable::Drawable;
use quick_renderer::gpu_immediate::GPUImmediate;
use quick_renderer::{glm, mesh};

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

pub trait MeshExtension<'de, END, EVD, EED, EFD> {
    type Error;

    fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, Self::Error>
    where
        Self: Sized;
    fn read_msgpack<P: AsRef<Path>>(path: P) -> Result<Self, Self::Error>
    where
        Self: Sized;

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

    fn visualize_config(&self, config: &Config<END, EVD, EED, EFD>, imm: &mut GPUImmediate) {
        match config.get_element() {
            crate::config::Element::Node => {
                // TODO(ish): handle showing which verts couldn't be
                // visualized
                self.visualize_node(config, imm);
            }
            crate::config::Element::Vert => todo!(),
            crate::config::Element::Edge => todo!(),
            crate::config::Element::Face => todo!(),
        }
    }
}

trait MeshExtensionPrivate<END, EVD, EED, EFD> {
    /// Tries to visualize all the links between the node and verts
    /// that refer to it.
    ///
    /// It returns back all the verts that cannot be visualized.
    fn visualize_node(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Vec<mesh::VertIndex>;
}

impl<END, EVD, EED, EFD> MeshExtensionPrivate<END, EVD, EED, EFD>
    for mesh::Mesh<END, EVD, EED, EFD>
{
    fn visualize_node(
        &self,
        config: &Config<END, EVD, EED, EFD>,
        imm: &mut GPUImmediate,
    ) -> Vec<mesh::VertIndex> {
        let mut no_uv_verts = Vec::new();
        let node = self
            .get_nodes()
            .get_unknown_gen(config.get_element_index())
            .unwrap()
            .0;
        node.get_verts().iter().for_each(|vert_index| {
            let vert = self.get_vert(*vert_index).unwrap();
            match vert.uv {
                Some(uv) => {
                    let uv_pos: glm::DVec3 = config
                        .get_uv_plane_3d_model_matrix()
                        .transform_vector(&glm::vec2_to_vec3(&uv));

                    let curve = CubicBezierCurve::new(
                        node.pos,
                        node.pos + node.normal.unwrap(),
                        uv_pos,
                        uv_pos,
                        20,
                    );

                    curve
                        .draw(&mut CubicBezierCurveDrawData::new(
                            imm,
                            glm::vec4(0.1, 0.4, 0.6, 1.0),
                        ))
                        .unwrap();
                }
                None => no_uv_verts.push(*vert_index),
            }
        });
        no_uv_verts
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
