use rmps::Deserializer;
use serde::Deserialize;

use std::path::Path;

use quick_renderer::mesh;

mod io_structs {
    use std::collections::HashMap;

    use quick_renderer::{glm, mesh};
    use serde::{Deserialize, Serialize};

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
            let mut mesh = Self::new();
            io_mesh.nodes.iter_mut().for_each(|io_node| {
                mesh.get_nodes_mut().insert_with(|self_index| {
                    let mut node = mesh::Node::new(mesh::NodeIndex(self_index), io_node.pos.into());
                    node.pos = io_node.pos.into();
                    node.normal = Some(io_node.normal.into());
                    node.extra_data = io_node.extra_data.take();

                    // TODO(ish): need to set verts

                    node
                });
            });

            io_mesh.verts.iter_mut().for_each(|io_vert| {
                mesh.get_verts_mut().insert_with(|self_index| {
                    let mut vert = mesh::Vert::new(mesh::VertIndex(self_index));

                    vert.uv = Some(io_vert.uv.into());
                    vert.extra_data = io_vert.extra_data.take();

                    // TODO(ish): need to set node and edges

                    vert
                });
            });

            io_mesh.edges.iter_mut().for_each(|io_edge| {
                mesh.get_edges_mut().insert_with(|self_index| {
                    let mut edge = mesh::Edge::new(mesh::EdgeIndex(self_index));

                    edge.extra_data = io_edge.extra_data.take();

                    // TODO(ish): need to set verts and faces
                    edge
                });
            });

            io_mesh.faces.iter_mut().for_each(|io_face| {
                mesh.get_faces_mut().insert_with(|self_index| {
                    let mut face = mesh::Face::new(mesh::FaceIndex(self_index));

                    face.normal = Some(io_face.normal.into());
                    face.extra_data = io_face.extra_data.take();

                    // TODO(ish): need to set verts

                    face
                });
            });

            assert_eq!(io_mesh.faces.len(), mesh.get_faces().len());
            assert_eq!(io_mesh.edges.len(), mesh.get_edges().len());
            assert_eq!(io_mesh.verts.len(), mesh.get_verts().len());
            assert_eq!(io_mesh.nodes.len(), mesh.get_nodes().len());

            mesh
        }
    }
}

pub trait MeshExtension {
    type Error;

    fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, Self::Error>
    where
        Self: Sized;
    fn read_msgpack<P: AsRef<Path>>(path: P) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

impl<END, EVD, EED, EFD> MeshExtension for mesh::Mesh<END, EVD, EED, EFD> {
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

        let meshio: io_structs::Mesh<(), (), (), ()> = Deserialize::deserialize(&mut de).unwrap();

        println!("meshio: {:?}", meshio);

        Ok(mesh::Mesh::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_read_file_msgpack() {
        mesh::simple::Mesh::read_file("/tmp/test.msgpack").unwrap();
    }
}
