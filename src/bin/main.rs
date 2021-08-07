use egui::{FontDefinitions, FontFamily, TextStyle};
use egui_glfw::EguiBackend;
use glfw::{Action, Context, Key};

use mesh_analyzer::blender_mesh_io::{AdaptiveMeshExtension, MeshExtensionError, MeshUVDrawData};
use quick_renderer::camera::WindowCamera;
use quick_renderer::drawable::Drawable;
use quick_renderer::fps::FPS;
use quick_renderer::gpu_immediate::GPUImmediate;
use quick_renderer::mesh::{MeshDrawData, MeshUseShader};
use quick_renderer::shader;
use quick_renderer::{egui, egui_glfw, gl, glfw, glm};

use mesh_analyzer::config::Config;
use mesh_analyzer::prelude::*;

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
        .create_window(1280, 720, "Mesh Analyzer", glfw::WindowMode::Windowed)
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
        gl::Enable(gl::BLEND);
        gl::Enable(gl::FRAMEBUFFER_SRGB);
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

    let config_file_path = "mesh_analyzer.config";
    let mut config = Config::from_file(config_file_path).unwrap_or_else(|_| {
        println!("No config found or error while loading config, using default");
        Config::default()
    });

    let mut fps = FPS::default();

    while !window.should_close() {
        glfw.poll_events();

        glfw::flush_messages(&events).for_each(|(_, event)| {
            egui.handle_event(&event, &window);

            handle_window_event(
                &event,
                &mut window,
                &mut camera,
                &mut config,
                &mut last_cursor,
            );
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

        let mesh_errors_maybe: Result<(), MeshExtensionError>;
        let adaptive_mesh_errors_maybe: Result<(), MeshExtensionError>;
        // Draw mesh
        {
            match config.get_mesh() {
                Ok(result_mesh) => match result_mesh {
                    Ok(mesh) => {
                        directional_light_shader.use_shader();
                        let model = glm::convert(config.get_mesh_transform().get_matrix());
                        directional_light_shader.set_mat4("model\0", &model);
                        mesh.draw(&mut MeshDrawData::new(
                            &mut imm,
                            MeshUseShader::DirectionalLight,
                            None,
                        ))
                        .unwrap();

                        mesh.draw_uv(&mut MeshUVDrawData::new(
                            &mut imm,
                            &glm::convert(config.get_uv_plane_3d_transform().get_matrix()),
                            &glm::convert(config.get_uv_map_color()),
                        ));

                        if config.get_draw_wireframe() {
                            smooth_color_3d_shader.use_shader();
                            smooth_color_3d_shader.set_mat4("model\0", &model);

                            mesh.draw_wireframe(&mut MeshDrawData::new(
                                &mut imm,
                                MeshUseShader::SmoothColor3D,
                                Some(glm::vec4(0.8, 0.8, 0.8, 1.0)),
                            ))
                            .unwrap();
                        }

                        smooth_color_3d_shader.use_shader();
                        smooth_color_3d_shader.set_mat4("model\0", &glm::identity());

                        mesh_errors_maybe = mesh.visualize_config(&config, &mut imm);
                        adaptive_mesh_errors_maybe =
                            mesh.adaptive_mesh_visualize_config(&config, &mut imm);
                    }
                    Err(_) => {
                        mesh_errors_maybe = Err(MeshExtensionError::NoMesh);
                        adaptive_mesh_errors_maybe = Ok(());
                    }
                },
                Err(error) => {
                    mesh_errors_maybe = Err(error);
                    adaptive_mesh_errors_maybe = Ok(());
                }
            }
        }

        // GUI starts
        {
            egui.begin_frame(&window, &mut glfw);
            egui::SidePanel::left("Left Side Panel")
                .resizable(true)
                .show(egui.get_egui_ctx(), |ui| {
                    egui::ScrollArea::auto_sized().show(ui, |ui| {
                        ui.label(format!("fps: {:.2}", fps.update_and_get()));
                        if ui.button("Save Config").clicked() {
                            let config_serialized = serde_json::to_string_pretty(&config).unwrap();
                            std::fs::write(config_file_path, config_serialized).unwrap();
                        }
                        if let Err(error) = mesh_errors_maybe {
                            ui.label(format!(
                                "Some error(s) while trying to visualize the mesh: {}",
                                error
                            ));
                        } else {
                            ui.label("Visualizing the mesh :)");
                        }
                        if let Err(error) = adaptive_mesh_errors_maybe {
                            // TODO(ish): This whole adaptive mesh
                            // visualization part is hacky, need to
                            // fix sometime
                            ui.label(format!(
                                "Some error(s) while trying to visualize the adaptive mesh: {}",
                                error
                            ));
                        } else if config.get_draw_anisotropic_flippable_edges() {
                            ui.label("Visualizing the adaptive mesh :)");
                        } else {
                            ui.label("Not visualzing the adaptive mesh part");
                        }
                        config.draw_ui(&(), ui);
                        config.draw_ui_edit(&(), ui);
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

fn handle_window_event<END, EVD, EED, EFD>(
    event: &glfw::WindowEvent,
    window: &mut glfw::Window,
    camera: &mut WindowCamera,
    config: &mut Config<END, EVD, EED, EFD>,
    last_cursor: &mut (f64, f64),
) {
    let cursor = window.get_cursor_pos();
    let mods;
    match event {
        glfw::WindowEvent::Key(_, _, _, modifiers) => {
            mods = Some(modifiers);
        }
        glfw::WindowEvent::MouseButton(_, _, modifiers) => mods = Some(modifiers),
        _ => {
            mods = None;
        }
    }
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

    if window.get_mouse_button(glfw::MouseButtonLeft) == glfw::Action::Press {
        if let Some(mods) = mods {
            if mods.contains(glfw::Modifiers::Control) {
                let ray_origin = camera.get_position();
                let ray_direction = camera.get_raycast_direction(cursor.0, cursor.1, window);

                config
                    .select_element((ray_origin, ray_direction))
                    .unwrap_or_default();
            }
        }
    }

    *last_cursor = cursor;
}
