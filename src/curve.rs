use std::fmt::Display;

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

/// Based on https://en.wikipedia.org/wiki/Point-normal_triangle and
/// https://alex.vlachos.com/graphics/CurvedPNTriangles.pdf
pub struct CubicPointNormalTriangle {
    pub p1: glm::DVec3,
    pub p2: glm::DVec3,
    pub p3: glm::DVec3,

    pub n1: glm::DVec3,
    pub n2: glm::DVec3,
    pub n3: glm::DVec3,

    pub num_steps: usize,
}

impl Default for CubicPointNormalTriangle {
    fn default() -> Self {
        Self {
            p1: Default::default(),
            p2: Default::default(),
            p3: Default::default(),
            n1: glm::vec3(0.0, 1.0, 0.0),
            n2: glm::vec3(0.0, 1.0, 0.0),
            n3: glm::vec3(0.0, 1.0, 0.0),
            num_steps: 1,
        }
    }
}

impl CubicPointNormalTriangle {
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

    /// Subdivides this PN triangle to form 9 PN triangles
    pub fn compute_one_level(&self) -> [CubicPointNormalTriangle; 9] {
        assert!(self.num_steps > 0);
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

        // TODO(ish): do the normal information stuff
        let normal = glm::vec3(1.0, 1.0, 1.0);

        [
            CubicPointNormalTriangle::new(
                b300,
                b210,
                b201,
                normal,
                normal,
                normal,
                self.num_steps - 1,
            ),
            CubicPointNormalTriangle::new(
                b210,
                b120,
                b111,
                normal,
                normal,
                normal,
                self.num_steps - 1,
            ),
            CubicPointNormalTriangle::new(
                b210,
                b111,
                b201,
                normal,
                normal,
                normal,
                self.num_steps - 1,
            ),
            CubicPointNormalTriangle::new(
                b201,
                b111,
                b102,
                normal,
                normal,
                normal,
                self.num_steps - 1,
            ),
            CubicPointNormalTriangle::new(
                b120,
                b030,
                b021,
                normal,
                normal,
                normal,
                self.num_steps - 1,
            ),
            CubicPointNormalTriangle::new(
                b120,
                b021,
                b111,
                normal,
                normal,
                normal,
                self.num_steps - 1,
            ),
            CubicPointNormalTriangle::new(
                b111,
                b021,
                b012,
                normal,
                normal,
                normal,
                self.num_steps - 1,
            ),
            CubicPointNormalTriangle::new(
                b111,
                b012,
                b102,
                normal,
                normal,
                normal,
                self.num_steps - 1,
            ),
            CubicPointNormalTriangle::new(
                b102,
                b012,
                b003,
                normal,
                normal,
                normal,
                self.num_steps - 1,
            ),
        ]
    }
}

pub struct CubicPointNormalTriangleDrawData<'a> {
    imm: &'a mut GPUImmediate,
    color: glm::Vec4,
}

impl<'a> CubicPointNormalTriangleDrawData<'a> {
    pub fn new(imm: &'a mut GPUImmediate, color: glm::Vec4) -> Self {
        Self { imm, color }
    }
}

impl Drawable<CubicPointNormalTriangleDrawData<'_>, Error> for CubicPointNormalTriangle {
    fn draw(&self, extra_data: &mut CubicPointNormalTriangleDrawData) -> Result<(), Error> {
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

        imm.begin(GPUPrimType::Tris, 9 * 3, &smooth_color_3d_shader);

        // TODO(ish): need to compute multiple levels

        let triangles = self.compute_one_level();

        triangles.iter().for_each(|triangle| {
            let p1: glm::Vec3 = glm::convert(triangle.p1);
            let p2: glm::Vec3 = glm::convert(triangle.p2);
            let p3: glm::Vec3 = glm::convert(triangle.p3);
            imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
            imm.vertex_3f(pos_attr, p1[0], p1[1], p1[2]);

            imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
            imm.vertex_3f(pos_attr, p2[0], p2[1], p2[2]);

            imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
            imm.vertex_3f(pos_attr, p3[0], p3[1], p3[2]);
        });

        imm.end();

        Ok(())
    }

    fn draw_wireframe(
        &self,
        _extra_data: &mut CubicPointNormalTriangleDrawData,
    ) -> Result<(), Error> {
        unreachable!()
    }
}
