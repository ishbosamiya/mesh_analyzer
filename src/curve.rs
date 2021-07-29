use std::fmt::Display;

use itertools::Itertools;
use quick_renderer::drawable::Drawable;
use quick_renderer::glm;
use quick_renderer::gpu_immediate::GPUImmediate;
use quick_renderer::gpu_immediate::GPUPrimType;
use quick_renderer::gpu_immediate::GPUVertCompType;
use quick_renderer::gpu_immediate::GPUVertFetchMode;
use quick_renderer::shader;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Error {
    InvalidNumberOfSteps,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidNumberOfSteps => write!(f, "Invalid Number of Steps"),
        }
    }
}

impl std::error::Error for Error {}

pub struct CubicBezierCurve {
    pub p0: glm::DVec3,
    pub p1: glm::DVec3,
    pub p2: glm::DVec3,
    pub p3: glm::DVec3,
    pub num_steps: usize,
}

impl Default for CubicBezierCurve {
    fn default() -> Self {
        Self {
            p0: Default::default(),
            p1: Default::default(),
            p2: Default::default(),
            p3: Default::default(),
            num_steps: 2,
        }
    }
}

impl CubicBezierCurve {
    pub fn new(
        p0: glm::DVec3,
        p1: glm::DVec3,
        p2: glm::DVec3,
        p3: glm::DVec3,
        num_steps: usize,
    ) -> Self {
        Self {
            p0,
            p1,
            p2,
            p3,
            num_steps,
        }
    }

    pub fn at_t(&self, t: f64) -> glm::DVec3 {
        (1.0 - t).powi(3) * self.p0
            + 3.0 * (1.0 - t).powi(2) * t * self.p1
            + 3.0 * (1.0 - t) * t.powi(2) * self.p2
            + t.powi(3) * self.p3
    }
}

pub struct CubicBezierCurveDrawData<'a> {
    imm: &'a mut GPUImmediate,
    color: glm::Vec4,
}

impl<'a> CubicBezierCurveDrawData<'a> {
    pub fn new(imm: &'a mut GPUImmediate, color: glm::Vec4) -> Self {
        Self { imm, color }
    }
}

impl Drawable<CubicBezierCurveDrawData<'_>, Error> for CubicBezierCurve {
    fn draw(&self, extra_data: &mut CubicBezierCurveDrawData) -> Result<(), Error> {
        if self.num_steps < 2 {
            return Err(Error::InvalidNumberOfSteps);
        }

        let imm = &mut extra_data.imm;
        let color = &extra_data.color;

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

        imm.begin(
            GPUPrimType::LineStrip,
            self.num_steps,
            &smooth_color_3d_shader,
        );

        (0..self.num_steps).for_each(|t| {
            let t = t as f64 * 1.0 / (self.num_steps - 1) as f64;
            let point = self.at_t(t);

            let point: glm::Vec3 = glm::convert(point);

            imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
            imm.vertex_3f(pos_attr, point[0], point[1], point[2]);
        });

        imm.end();

        Ok(())
    }

    fn draw_wireframe(&self, _extra_data: &mut CubicBezierCurveDrawData) -> Result<(), Error> {
        unreachable!()
    }
}

struct ControlPoints {
    positions: [glm::DVec3; 10],
    normals: [glm::DVec3; 6],
}

impl ControlPoints {
    fn new(positions: [glm::DVec3; 10], normals: [glm::DVec3; 6]) -> Self {
        Self { positions, normals }
    }

    #[allow(clippy::zero_prefixed_literal)]
    fn get_control_point(&self, number: usize) -> &glm::DVec3 {
        &self.positions[match number {
            300 => 0,
            030 => 1,
            003 => 2,
            210 => 3,
            120 => 4,
            021 => 5,
            012 => 6,
            102 => 7,
            201 => 8,
            111 => 9,
            _ => unreachable!(),
        }]
    }

    #[allow(clippy::zero_prefixed_literal)]
    fn get_normal_point(&self, number: usize) -> &glm::DVec3 {
        &self.normals[match number {
            200 => 0,
            020 => 1,
            002 => 2,
            110 => 3,
            011 => 4,
            101 => 5,
            _ => unreachable!(),
        }]
    }

    #[allow(clippy::zero_prefixed_literal)]
    fn get_pos_at(&self, u: f64, v: f64) -> glm::DVec3 {
        let w = 1.0 - u - v;
        self.get_control_point(300) * w.powi(3)
            + self.get_control_point(030) * u.powi(3)
            + self.get_control_point(003) * v.powi(3)
            + self.get_control_point(210) * 3.0 * w.powi(2) * u
            + self.get_control_point(120) * 3.0 * w * u.powi(2)
            + self.get_control_point(201) * 3.0 * w.powi(2) * v
            + self.get_control_point(021) * 3.0 * u.powi(2) * v
            + self.get_control_point(102) * 3.0 * w * v.powi(2)
            + self.get_control_point(012) * 3.0 * u * v.powi(2)
            + self.get_control_point(111) * 6.0 * w * u * v
    }

    #[allow(clippy::zero_prefixed_literal)]
    fn get_normal_at(&self, u: f64, v: f64) -> glm::DVec3 {
        let w = 1.0 - u - v;
        self.get_normal_point(200) * w.powi(2)
            + self.get_normal_point(020) * u.powi(2)
            + self.get_normal_point(002) * v.powi(2)
            + self.get_normal_point(110) * w * u
            + self.get_normal_point(011) * u * v
            + self.get_normal_point(101) * w * v
    }
}

/// Based on https://en.wikipedia.org/wiki/Point-normal_triangle and
/// https://alex.vlachos.com/graphics/CurvedPNTriangles.pdf
#[derive(Debug, Clone)]
pub struct PointNormalTriangle {
    pub p1: glm::DVec3,
    pub p2: glm::DVec3,
    pub p3: glm::DVec3,

    pub n1: glm::DVec3,
    pub n2: glm::DVec3,
    pub n3: glm::DVec3,

    pub num_steps: usize,
}

impl Default for PointNormalTriangle {
    fn default() -> Self {
        Self::new(
            glm::vec3(1.0, 0.0, 0.0),
            glm::vec3(-1.0, 0.0, 0.0),
            glm::vec3(0.0, 0.0, 1.0),
            glm::vec3(0.0, 1.0, 0.0),
            glm::vec3(0.0, 1.0, 0.0),
            glm::vec3(0.0, 1.0, 0.0),
            1,
        )
    }
}

impl PointNormalTriangle {
    pub fn new(
        p1: glm::DVec3,
        p2: glm::DVec3,
        p3: glm::DVec3,
        n1: glm::DVec3,
        n2: glm::DVec3,
        n3: glm::DVec3,
        num_steps: usize,
    ) -> Self {
        Self {
            p1,
            p2,
            p3,
            n1,
            n2,
            n3,
            num_steps,
        }
    }

    fn get_control_points(&self) -> ControlPoints {
        let p1 = &self.p1;
        let p2 = &self.p2;
        let p3 = &self.p3;
        let n1 = &self.n1;
        let n2 = &self.n2;
        let n3 = &self.n3;

        let b300 = *p1;
        let b030 = *p2;
        let b003 = *p3;

        let compute_w =
            |pi: &glm::DVec3, pj: &glm::DVec3, ni: &glm::DVec3| glm::dot(&(pj - pi), ni);

        let b210 = (2.0 * p1 + p2 - compute_w(p1, p2, n1) * n1) / 3.0;
        let b120 = (2.0 * p2 + p1 - compute_w(p2, p1, n2) * n2) / 3.0;
        let b021 = (2.0 * p2 + p3 - compute_w(p2, p3, n2) * n2) / 3.0;
        let b012 = (2.0 * p3 + p2 - compute_w(p3, p2, n3) * n3) / 3.0;
        let b102 = (2.0 * p3 + p1 - compute_w(p3, p1, n3) * n3) / 3.0;
        let b201 = (2.0 * p1 + p3 - compute_w(p1, p3, n1) * n1) / 3.0;

        let e = (b210 + b120 + b021 + b012 + b102 + b201) / 6.0;
        let v = (p1 + p2 + p3) / 3.0;
        let b111 = e + (e - v) / 2.0;

        let n200 = *n1;
        let n020 = *n2;
        let n002 = *n3;

        let compute_v = |pi: &glm::DVec3, pj: &glm::DVec3, ni: &glm::DVec3, nj: &glm::DVec3| {
            2.0 * glm::dot(&(pj - pi), &(ni + nj)) / glm::dot(&(pj - pi), &(pj - pi))
        };

        let h110 = n1 + n2 + compute_v(p1, p2, n1, n2) * (p2 - p1);
        let h011 = n2 + n3 + compute_v(p2, p3, n2, n3) * (p3 - p2);
        let h101 = n3 + n1 + compute_v(p3, p1, n3, n1) * (p1 - p3);

        let n110 = h110.normalize();
        let n011 = h011.normalize();
        let n101 = h101.normalize();

        ControlPoints::new(
            [b300, b030, b003, b210, b120, b021, b012, b102, b201, b111],
            [n200, n020, n002, n110, n011, n101],
        )
    }
}

pub struct PointNormalTriangleDrawData<'a> {
    imm: &'a mut GPUImmediate,
    color: glm::Vec4,
    display_vertex_normals: bool,
    normal_factor: f64,
}

impl<'a> PointNormalTriangleDrawData<'a> {
    pub fn new(
        imm: &'a mut GPUImmediate,
        color: glm::Vec4,
        display_vertex_normals: bool,
        normal_factor: f64,
    ) -> Self {
        Self {
            imm,
            color,
            display_vertex_normals,
            normal_factor,
        }
    }
}

impl Drawable<PointNormalTriangleDrawData<'_>, Error> for PointNormalTriangle {
    fn draw(&self, extra_data: &mut PointNormalTriangleDrawData) -> Result<(), Error> {
        if self.num_steps == 0 {
            return Err(Error::InvalidNumberOfSteps);
        }

        let imm = &mut extra_data.imm;
        let color = &extra_data.color;

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

        let control_points = self.get_control_points();

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // generate the verts
        (0..=self.num_steps).for_each(|i| {
            let u = i as f64 / (self.num_steps) as f64;
            (0..=self.num_steps).for_each(|j| {
                let v = j as f64 / (self.num_steps) as f64;

                let pos = control_points.get_pos_at(u, v);
                let normal = control_points.get_normal_at(u, v);

                let pos: glm::Vec3 = glm::convert(pos);
                let normal: glm::Vec3 = glm::convert(normal);

                vertices.push((u, v, pos, normal));
            });
        });

        // generate the triangle indices
        (0..self.num_steps).for_each(|i| {
            (0..self.num_steps).for_each(|j| {
                indices.push((self.num_steps + 1) * j + i);
                indices.push((self.num_steps + 1) * (j + 1) + i);
                indices.push((self.num_steps + 1) * j + i + 1);

                indices.push((self.num_steps + 1) * j + i + 1);
                indices.push((self.num_steps + 1) * (j + 1) + i);
                indices.push((self.num_steps + 1) * (j + 1) + i + 1);
            });
        });

        assert!(!indices.is_empty());
        assert!(indices.len() % 3 == 0);

        imm.begin_at_most(GPUPrimType::Tris, indices.len(), &smooth_color_3d_shader);

        for chunk in &indices.iter().chunks(3) {
            let mut use_this = true;

            let chunk: Vec<_> = chunk
                .map(|index| {
                    let (u, v, _, _) = vertices[*index];
                    if u + v > 1.0 {
                        use_this = false;
                        return usize::MAX;
                    }
                    *index
                })
                .collect();

            if use_this {
                chunk.iter().for_each(|index| {
                    let (_, _, pos, _) = vertices[*index];

                    imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
                    imm.vertex_3f(pos_attr, pos[0], pos[1], pos[2]);
                });
            }
        }

        imm.end();

        let normal_factor = extra_data.normal_factor;

        if extra_data.display_vertex_normals {
            imm.begin_at_most(
                GPUPrimType::Lines,
                vertices.len() * 2,
                &smooth_color_3d_shader,
            );

            vertices.iter().for_each(|(u, v, pos, normal)| {
                if u + v > 1.0 {
                    return;
                }

                let pos_2: glm::Vec3 = pos + normal * normal_factor as f32;

                imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
                imm.vertex_3f(pos_attr, pos[0], pos[1], pos[2]);

                imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
                imm.vertex_3f(pos_attr, pos_2[0], pos_2[1], pos_2[2]);
            });

            imm.end();
        }

        Ok(())
    }

    fn draw_wireframe(&self, _extra_data: &mut PointNormalTriangleDrawData) -> Result<(), Error> {
        unreachable!()
    }
}

/// Given p1, p2, p3 that form the triangle, and p is the point for
/// which the barycentric coords must be found.
///
/// # Reference
/// https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
fn _calculate_barycentric_coords(
    p1: &glm::DVec3,
    p2: &glm::DVec3,
    p3: &glm::DVec3,
    p: &glm::DVec3,
) -> glm::DVec3 {
    let v0 = p2 - p1;
    let v1 = p3 - p1;
    let v2 = p - p1;

    let d00 = glm::dot(&v0, &v0);
    let d01 = glm::dot(&v0, &v1);
    let d11 = glm::dot(&v1, &v1);
    let d20 = glm::dot(&v2, &v0);
    let d21 = glm::dot(&v2, &v1);

    #[allow(clippy::suspicious_operation_groupings)]
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    glm::vec3(u, v, w)
}

fn _apply_barycentric_coords(
    bary_coords: &glm::DVec3,
    p1: &glm::DVec3,
    p2: &glm::DVec3,
    p3: &glm::DVec3,
) -> glm::DVec3 {
    bary_coords[0] * p1 + bary_coords[1] * p2 + bary_coords[2] * p3
}
