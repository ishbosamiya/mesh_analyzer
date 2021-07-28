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
