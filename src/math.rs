use quick_renderer::glm;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Transform {
    pub location: glm::DVec3,
    pub rotation: glm::DVec3,
    pub scale: glm::DVec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            location: glm::zero(),
            rotation: glm::zero(),
            scale: glm::vec3(1.0, 1.0, 1.0),
        }
    }
}

impl Transform {
    pub fn new(location: glm::DVec3, rotation: glm::DVec3, scale: glm::DVec3) -> Self {
        Self {
            location,
            rotation,
            scale,
        }
    }

    pub fn get_matrix(&self) -> glm::DMat4 {
        let scaled_mat = glm::scale(&glm::identity(), &self.scale);
        let rotated_mat = glm::rotate_z(
            &glm::rotate_y(
                &glm::rotate_x(&scaled_mat, self.rotation[0]),
                self.rotation[1],
            ),
            self.rotation[2],
        );

        glm::translate(&rotated_mat, &self.location)
    }
}
