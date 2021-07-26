use egui::{FontDefinitions, FontFamily, TextStyle};
use egui_glfw::EguiBackend;
use glfw::{Action, Context, Key};

use quick_renderer::camera::WindowCamera;
use quick_renderer::drawable::Drawable;
use quick_renderer::gpu_immediate::GPUImmediate;
use quick_renderer::mesh::simple::Mesh;
use quick_renderer::mesh::MeshDrawData;
use quick_renderer::meshreader::MeshReader;
use quick_renderer::shader;
use quick_renderer::{egui, egui_glfw, gl, glfw, glm};

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

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

    let cube = r#"
# Blender v2.92.0 Alpha OBJ File: ''
# www.blender.org
o Cube_Cube.001
v -1.000000 -1.000000 1.000000
v -1.000000 1.000000 1.000000
v -1.000000 -1.000000 -1.000000
v -1.000000 1.000000 -1.000000
v 1.000000 -1.000000 1.000000
v 1.000000 1.000000 1.000000
v 1.000000 -1.000000 -1.000000
v 1.000000 1.000000 -1.000000
vt 0.375000 0.000000
vt 0.625000 0.000000
vt 0.625000 0.250000
vt 0.375000 0.250000
vt 0.625000 0.500000
vt 0.375000 0.500000
vt 0.625000 0.750000
vt 0.375000 0.750000
vt 0.625000 1.000000
vt 0.375000 1.000000
vt 0.125000 0.500000
vt 0.125000 0.750000
vt 0.875000 0.500000
vt 0.875000 0.750000
vn -1.0000 0.0000 0.0000
vn 0.0000 0.0000 -1.0000
vn 1.0000 0.0000 0.0000
vn 0.0000 0.0000 1.0000
vn 0.0000 -1.0000 0.0000
vn 0.0000 1.0000 0.0000
s off
f 1/1/1 2/2/1 4/3/1 3/4/1
f 3/4/2 4/3/2 8/5/2 7/6/2
f 7/6/3 8/5/3 6/7/3 5/8/3
f 5/8/4 6/7/4 2/9/4 1/10/4
f 3/11/5 7/6/5 5/8/5 1/12/5
f 8/5/6 4/13/6 2/14/6 6/7/6
"#;
    let cube_lines: Vec<&str> = cube.split('\n').collect();
    let meshreader = MeshReader::from_lines(&cube_lines).unwrap();
    let mesh = Mesh::read(&meshreader).unwrap();

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
    println!(
        "directional_light: uniforms: {:?} attributes: {:?}",
        directional_light_shader.get_uniforms(),
        directional_light_shader.get_attributes(),
    );

    let mut last_cursor = window.get_cursor_pos();

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
            directional_light_shader.use_shader();
            directional_light_shader.set_mat4(
                "projection\0",
                &glm::convert(camera.get_projection_matrix(&window)),
            );
            directional_light_shader.set_mat4("view\0", &glm::convert(camera.get_view_matrix()));
            directional_light_shader.set_mat4("model\0", &glm::identity());
            directional_light_shader.set_vec3("viewPos\0", &glm::convert(camera.get_position()));
            directional_light_shader.set_vec3("material.color\0", &glm::vec3(0.3, 0.2, 0.7));
            directional_light_shader.set_vec3("material.specular\0", &glm::vec3(0.3, 0.3, 0.3));
            directional_light_shader.set_float("material.shininess\0", 4.0);
            directional_light_shader.set_vec3("light.direction\0", &glm::vec3(-0.7, -1.0, -0.7));
            directional_light_shader.set_vec3("light.ambient\0", &glm::vec3(0.3, 0.3, 0.3));
            directional_light_shader.set_vec3("light.diffuse\0", &glm::vec3(1.0, 1.0, 1.0));
            directional_light_shader.set_vec3("light.specular\0", &glm::vec3(1.0, 1.0, 1.0));
        }

        directional_light_shader.use_shader();
        mesh.draw(&mut MeshDrawData::new(&mut imm, &directional_light_shader))
            .unwrap();

        // GUI starts
        {
            egui.begin_frame(&window, &mut glfw);
            egui::Window::new("Hello world!").show(egui.get_egui_ctx(), |ui| {
                ui.label("Hello World, Simple Render!");
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
