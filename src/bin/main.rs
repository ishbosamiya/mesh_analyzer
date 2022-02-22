use egui::{FontDefinitions, FontFamily, TextStyle};
use egui_glfw::EguiBackend;
use glfw::{Action, Context, Key};

use std::convert::TryInto;

#[cfg(feature = "use_cloth_adaptive_mesh")]
use mesh_analyzer::blender_mesh_io::ClothAdaptiveMeshExtension;
use mesh_analyzer::blender_mesh_io::{AdaptiveMeshExtension, MeshExtensionError, MeshUVDrawData};
use quick_renderer::camera::WindowCamera;
use quick_renderer::drawable::Drawable;
use quick_renderer::fps::FPS;
use quick_renderer::gpu_immediate::GPUImmediate;
use quick_renderer::infinite_grid::{InfiniteGrid, InfiniteGridDrawData};
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

    shader::builtins::display_uniform_and_attribute_info();
    let directional_light_shader = shader::builtins::get_directional_light_shader()
        .as_ref()
        .unwrap();

    let smooth_color_3d_shader = shader::builtins::get_smooth_color_3d_shader()
        .as_ref()
        .unwrap();

    let face_orientation_shader = shader::builtins::get_face_orientation_shader()
        .as_ref()
        .unwrap();

    unsafe {
        gl::PointSize(10.0);
    }

    let mut last_cursor = window.get_cursor_pos();

    let config_file_path = "mesh_analyzer.config";
    let mut config = Config::from_file(config_file_path).unwrap_or_else(|_| {
        println!("No config found or error while loading config, using default");
        Config::default()
    });

    let mut fps = FPS::default();

    let infinite_grid = InfiniteGrid::default();

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

        let (window_width, window_height) = window.get_size();
        let (window_width, window_height): (usize, usize) = (
            window_width.try_into().unwrap(),
            window_height.try_into().unwrap(),
        );

        // Shader stuff
        shader::builtins::setup_shaders(&camera, window_width, window_height);

        unsafe {
            gl::Disable(gl::BLEND);
        }

        let mesh_errors_maybe: Result<(), MeshExtensionError>;
        let adaptive_mesh_errors_maybe: Result<(), MeshExtensionError>;
        // Draw mesh
        {
            match config.get_mesh() {
                Ok(result_mesh) => match result_mesh {
                    Ok(mesh) => {
                        let model = glm::convert(config.get_mesh_transform().get_matrix());
                        match config.get_draw_mesh_with_shader() {
                            MeshUseShader::DirectionalLight => {
                                directional_light_shader.use_shader();
                                directional_light_shader.set_mat4("model\0", &model);
                                mesh.draw(&mut MeshDrawData::new(
                                    &mut imm,
                                    MeshUseShader::DirectionalLight,
                                    None,
                                ))
                                .unwrap();
                            }
                            MeshUseShader::FaceOrientation => {
                                face_orientation_shader.use_shader();
                                face_orientation_shader.set_mat4("model\0", &model);
                                mesh.draw(&mut MeshDrawData::new(
                                    &mut imm,
                                    MeshUseShader::FaceOrientation,
                                    None,
                                ))
                                .unwrap();
                            }
                            MeshUseShader::SmoothColor3D => unreachable!(),
                        }

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

                        #[cfg(feature = "use_cloth_adaptive_mesh")]
                        mesh.cloth_adaptive_mesh_visualize_config(&config, &mut imm)
                            .unwrap_or(());
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

        // Keep meshes that have shaders that need alpha channel
        // (blending) bellow this and handle it properly
        {
            unsafe {
                gl::Enable(gl::BLEND);
                gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            }
            {
                // TODO(ish): this is a hack to make the grid show up
                // better, it is too bright and lines are way too
                // thick when the framebuffer is in srgb. Really need
                // to spend the time to understand color related
                // things better so things can be not hacky and
                // visually pleasing
                unsafe {
                    gl::Disable(gl::FRAMEBUFFER_SRGB);
                }
                if config.get_draw_infinite_grid() {
                    infinite_grid
                        .draw(&mut InfiniteGridDrawData::new(&mut imm))
                        .unwrap();
                }
                unsafe {
                    gl::Enable(gl::FRAMEBUFFER_SRGB);
                }
            }
        }

        // GUI starts
        {
            egui.begin_frame(&window, &mut glfw);
            egui::SidePanel::left("Left Side Panel")
                .resizable(true)
                .show(egui.get_egui_ctx(), |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.label(format!("fps: {:.2}", fps.get_last_processed()));
                        if ui.button("Save Config").clicked() {
                            let mut config_backup_file_path =
                                std::path::PathBuf::from(config_file_path);
                            config_backup_file_path.set_extension("config~");
                            std::fs::rename(config_file_path, config_backup_file_path).unwrap();
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

                        #[cfg(feature = "use_cloth_adaptive_mesh")]
                        if let Ok(Ok(mesh)) = config.get_mesh() {
                            mesh.draw_ui(&config, ui);
                        }
                    });
                });
            let (width, height) = window.get_framebuffer_size();
            let _output = egui.end_frame(glm::vec2(width as _, height as _));
        }
        // GUI ends

        // fps update
        fps.update_and_get(Some(config.get_fps_limit()));

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

    let (window_width, window_height) = window.get_size();
    let (window_width, window_height): (usize, usize) = (
        window_width.try_into().unwrap(),
        window_height.try_into().unwrap(),
    );

    if window.get_mouse_button(glfw::MouseButtonMiddle) == glfw::Action::Press {
        if window.get_key(glfw::Key::LeftShift) == glfw::Action::Press {
            camera.pan(
                last_cursor.0,
                last_cursor.1,
                cursor.0,
                cursor.1,
                1.0,
                window_width,
                window_height,
            );
        } else if window.get_key(glfw::Key::LeftControl) == glfw::Action::Press {
            camera.move_forward(last_cursor.1, cursor.1, window_height);
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
                let ray_direction =
                    camera.get_raycast_direction(cursor.0, cursor.1, window_width, window_height);

                config
                    .select_element((ray_origin, ray_direction))
                    .unwrap_or_default();
            }
        }
    }

    *last_cursor = cursor;
}
