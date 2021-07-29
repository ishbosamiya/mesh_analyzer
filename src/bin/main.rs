use egui::{FontDefinitions, FontFamily, TextStyle};
use egui_glfw::EguiBackend;
use glfw::{Action, Context, Key};

use mesh_analyzer::blender_mesh_io::MeshUVDrawData;
use mesh_analyzer::curve::{PointNormalTriangle, PointNormalTriangleDrawData};
use quick_renderer::camera::WindowCamera;
use quick_renderer::drawable::Drawable;
use quick_renderer::gpu_immediate::GPUImmediate;
use quick_renderer::mesh::simple::Mesh;
use quick_renderer::mesh::MeshDrawData;
use quick_renderer::shader;
use quick_renderer::{egui, egui_glfw, gl, glfw, glm};

use mesh_analyzer::config::Config;
use mesh_analyzer::{prelude::*, ui_widgets};

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));
    glfw.window_hint(glfw::WindowHint::Samples(Some(16)));
    #[cfg(target_os = "macos")]
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));

    // creating window
    let (mut window, events) = glfw
        .create_window(1280, 720, "Simple Render", glfw::WindowMode::Windowed)
        .expect("ERROR: glfw window creation failed");

    // setup bunch of polling data
    window.set_key_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_mouse_button_polling(true);
    window.set_framebuffer_size_polling(true);
    window.set_scroll_polling(true);
    window.set_char_polling(true);
    window.make_current();

    gl::load_with(|symbol| window.get_proc_address(symbol));

    unsafe {
        gl::Disable(gl::CULL_FACE);
        gl::Enable(gl::DEPTH_TEST);
        gl::Enable(gl::MULTISAMPLE);
    }

    // setup the egui backend
    let mut egui = EguiBackend::new(&mut window, &mut glfw);

    let mut fonts = FontDefinitions::default();
    // larger text
    fonts
        .family_and_size
        .insert(TextStyle::Button, (FontFamily::Proportional, 18.0));
    fonts
        .family_and_size
        .insert(TextStyle::Body, (FontFamily::Proportional, 18.0));
    fonts
        .family_and_size
        .insert(TextStyle::Small, (FontFamily::Proportional, 15.0));
    egui.get_egui_ctx().set_fonts(fonts);

    let mesh = Mesh::read_file("/tmp/test.msgpack").unwrap();

    let mut camera = WindowCamera::new(
        glm::vec3(0.0, 0.0, 3.0),
        glm::vec3(0.0, 1.0, 0.0),
        -90.0,
        0.0,
        45.0,
    );

    let mut imm = GPUImmediate::new();

    let directional_light_shader = shader::builtins::get_directional_light_shader()
        .as_ref()
        .unwrap();

    let smooth_color_3d_shader = shader::builtins::get_smooth_color_3d_shader()
        .as_ref()
        .unwrap();

    println!(
        "directional_light: uniforms: {:?} attributes: {:?}",
        directional_light_shader.get_uniforms(),
        directional_light_shader.get_attributes(),
    );
    println!(
        "smooth_color_3d: uniforms: {:?} attributes: {:?}",
        smooth_color_3d_shader.get_uniforms(),
        smooth_color_3d_shader.get_attributes(),
    );

    let mut last_cursor = window.get_cursor_pos();

    let mut config = Config::default();

    let mut pn_triangle = PointNormalTriangle::default();
    let mut pn_triangle_display_normals = false;
    let mut pn_triangle_normal_factor = 0.1;

    while !window.should_close() {
        glfw.poll_events();

        glfw::flush_messages(&events).for_each(|(_, event)| {
            egui.handle_event(&event, &window);

            handle_window_event(&event, &mut window, &mut camera, &mut last_cursor);
        });

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }

        // Shader stuff
        {
            let projection_matrix = glm::convert(camera.get_projection_matrix(&window));
            let view_matrix = glm::convert(camera.get_view_matrix());
            {
                directional_light_shader.use_shader();
                directional_light_shader.set_mat4("projection\0", &projection_matrix);
                directional_light_shader.set_mat4("view\0", &view_matrix);
                directional_light_shader.set_mat4("model\0", &glm::identity());
                directional_light_shader
                    .set_vec3("viewPos\0", &glm::convert(camera.get_position()));
                directional_light_shader.set_vec3("material.color\0", &glm::vec3(0.3, 0.2, 0.7));
                directional_light_shader.set_vec3("material.specular\0", &glm::vec3(0.3, 0.3, 0.3));
                directional_light_shader.set_float("material.shininess\0", 4.0);
                directional_light_shader
                    .set_vec3("light.direction\0", &glm::vec3(-0.7, -1.0, -0.7));
                directional_light_shader.set_vec3("light.ambient\0", &glm::vec3(0.3, 0.3, 0.3));
                directional_light_shader.set_vec3("light.diffuse\0", &glm::vec3(1.0, 1.0, 1.0));
                directional_light_shader.set_vec3("light.specular\0", &glm::vec3(1.0, 1.0, 1.0));
            }

            {
                smooth_color_3d_shader.use_shader();
                smooth_color_3d_shader.set_mat4("projection\0", &projection_matrix);
                smooth_color_3d_shader.set_mat4("view\0", &view_matrix);
                smooth_color_3d_shader.set_mat4("model\0", &glm::identity());
            }
        }

        // Draw mesh
        {
            directional_light_shader.use_shader();

            let model = glm::convert(config.get_mesh_transform().get_matrix());
            directional_light_shader.set_mat4("model\0", &model);

            mesh.draw(&mut MeshDrawData::new(&mut imm, &directional_light_shader))
                .unwrap();

            mesh.draw_uv(&mut MeshUVDrawData::new(
                &mut imm,
                &glm::convert(config.get_uv_plane_3d_model_matrix()),
                &glm::convert(config.get_uv_map_color()),
            ));

            smooth_color_3d_shader.use_shader();
            smooth_color_3d_shader.set_mat4("model\0", &glm::identity());

            mesh.visualize_config(&config, &mut imm);

            pn_triangle
                .draw(&mut PointNormalTriangleDrawData::new(
                    &mut imm,
                    glm::vec4(0.1, 0.8, 0.8, 0.7),
                    pn_triangle_display_normals,
                    pn_triangle_normal_factor,
                    glm::vec4(0.8, 0.24, 0.2, 1.0),
                ))
                .unwrap();
        }

        // GUI starts
        {
            egui.begin_frame(&window, &mut glfw);
            egui::SidePanel::left("Left Side Panel")
                .resizable(true)
                .show(egui.get_egui_ctx(), |ui| {
                    egui::ScrollArea::auto_sized().show(ui, |ui| {
                        config.draw_ui(&mesh, ui);
                        config.draw_ui_edit(&mesh, ui);

                        ui.checkbox(&mut pn_triangle_display_normals, "Display Vertex Normals");
                        ui.label("Normal Factor");
                        ui.add(egui::Slider::new(&mut pn_triangle_normal_factor, 0.0..=1.0));
                        ui.label("Num Steps");
                        ui.add(egui::Slider::new(&mut pn_triangle.num_steps, 1..=25));
                        ui.label("p1");
                        ui.add(ui_widgets::Vec3::new(&mut pn_triangle.p1));
                        ui.label("p2");
                        ui.add(ui_widgets::Vec3::new(&mut pn_triangle.p2));
                        ui.label("p3");
                        ui.add(ui_widgets::Vec3::new(&mut pn_triangle.p3));
                        ui.label("n1");
                        ui.add(ui_widgets::Vec3::new(&mut pn_triangle.n1));
                        if ui.button("normalize").clicked() {
                            pn_triangle.n1.normalize_mut();
                        }
                        ui.label("n2");
                        ui.add(ui_widgets::Vec3::new(&mut pn_triangle.n2));
                        if ui.button("normalize").clicked() {
                            pn_triangle.n2.normalize_mut();
                        }
                        ui.label("n3");
                        ui.add(ui_widgets::Vec3::new(&mut pn_triangle.n3));
                        if ui.button("normalize").clicked() {
                            pn_triangle.n3.normalize_mut();
                        }
                    });
                });
            let (width, height) = window.get_framebuffer_size();
            let _output = egui.end_frame(glm::vec2(width as _, height as _));
        }
        // GUI ends

        // Swap front and back buffers
        window.swap_buffers();
    }
}

fn handle_window_event(
    event: &glfw::WindowEvent,
    window: &mut glfw::Window,
    camera: &mut WindowCamera,
    last_cursor: &mut (f64, f64),
) {
    let cursor = window.get_cursor_pos();
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true);
        }

        glfw::WindowEvent::FramebufferSize(width, height) => unsafe {
            gl::Viewport(0, 0, *width, *height);
        },
        glfw::WindowEvent::Scroll(_, scroll_y) => {
            camera.zoom(*scroll_y);
        }
        _ => {}
    };

    if window.get_mouse_button(glfw::MouseButtonMiddle) == glfw::Action::Press {
        if window.get_key(glfw::Key::LeftShift) == glfw::Action::Press {
            camera.pan(
                last_cursor.0,
                last_cursor.1,
                cursor.0,
                cursor.1,
                1.0,
                window,
            );
        } else if window.get_key(glfw::Key::LeftControl) == glfw::Action::Press {
            camera.move_forward(last_cursor.1, cursor.1, window);
        } else {
            camera.rotate_wrt_camera_origin(
                last_cursor.0,
                last_cursor.1,
                cursor.0,
                cursor.1,
                0.1,
                false,
            );
        }
    }

    *last_cursor = cursor;
}
