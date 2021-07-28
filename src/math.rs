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
        let translated_mat = glm::translate(&glm::identity(), &self.location);
        let rotated_mat = glm::rotate_z(
            &glm::rotate_y(
                &glm::rotate_x(&translated_mat, self.rotation[0]),
                self.rotation[1],
            ),
            self.rotation[2],
        );
        let scaled_mat = glm::scale(&rotated_mat, &self.scale);

        scaled_mat
    }
}
