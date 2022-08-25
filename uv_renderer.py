import moderngl
from PIL import Image
from pyrr import Matrix44
import numpy as np


class UVRenderer:

    def __init__(self, obj_3d, viewport=(299, 299)):
        """
        Construct a Renderer object used to compute the UV mapping for a certain 3D model, in a certain pose with a
        random rotation, translation, camera distance, background, photo error and texture printing error. These random
        variables are drawn from bounded or unbounded uniform distributions.

        Parameters
        ----------
        obj_3d : objloader.Obj
            Obj object representing the 3D model.
        viewport : tuple(int, int)
            width and height of the rendered image.
        """
        self.width, self.height = viewport
        self.object = obj_3d

        # Create ModernGL context, which exposes OpenGL features. Require OpenGL 330 core profile
        self.ctx = moderngl.create_standalone_context(require=330)
        # use depth test and face culling to only render parts on the model visible in the image
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        # Create frame buffer, where the scene will be rendered. Texture is empty, as we do not need the actual texture
        # of the model for calculating the UV mapping.
        self.fbo = self.ctx.framebuffer(
            [self.ctx.texture(viewport, components=2, dtype='f4')],
            self.ctx.depth_renderbuffer(viewport)
        )

        # shader program
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330 core

                // model view projection matrix
                uniform mat4 mvp;

                in vec3 in_vert;
                in vec2 in_text;

                out vec2 v_text;

                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0f);
                    // UV coordinate starts from bottom left corner
                    // image coordinate starts from top left corner
                    // therefore we need to flip the V-axis
                    v_text = vec2(in_text.x, 1.0f - in_text.y);
                }
            ''',
            # since we do not render the texture, the fragment shader does nothing
            fragment_shader='''
                #version 330 core

                in vec2 v_text;

                out vec2 f_color;

                void main() {
                    f_color = v_text;
                }
            '''
        )

        # get reference to uniform pointing to Model View Projection Matrix
        self.mvp = self.prog["mvp"]
        self.vao = []


    def set_parameters(self,
                       camera_distance=(2.5, 3.0),
                       x_translation=(-0.05, 0.05),
                       y_translation=(-0.05, 0.05),
                       deflection=1.0):
        """
        Set parameters for rendering.

        Parameters
        ----------
        camera_distance : tuple(float, float)
            The minimum and maximum distance from camera.
        x_translation : tuple(float, float)
            The minimum and maximum translation along x-axis.
        y_translation : tuple(float, float)
            The minimum and maximum translation along y-axis.
        deflection : float between 0 and 1.
            The magnitude of the rotation, see rand_rotation_matrix.
        """
        self.close, self.far = camera_distance
        self.x_low, self.x_high = x_translation
        self.y_low, self.y_high = y_translation

        assert 0 <= deflection <= 1
        self.deflection = deflection

    def set_obj(self, new_obj):
        self.object = new_obj

    @staticmethod
    def rand_rotation_matrix(deflection=1.0, randnums=None):
        """
        Creates a random rotation matrix.

        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
        """
        if randnums is None:
            randnums = np.random.uniform(size=(3,))

        theta, phi, z = randnums

        # Rotation about the pole (Z).
        theta = theta * 2.0 * deflection * np.pi
        phi = phi * 2.0 * np.pi  # For direction of pole deflection.
        z = z * 2.0 * deflection  # For magnitude of pole deflection.

        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.

        r = np.sqrt(z)
        V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
        )
        Vx, Vy, Vz = V

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.
        M = (np.outer(V, V) - np.eye(3)).dot(R)

        return M

    def render(self, i=0, save_render=False):
        """
        Render a batch of images of the obj_3d, each time in a different random pose, and returns the UV mappings for
        each.

        Parameters
        ----------
        i : int
            Number of uv mapping in batch. Used when naming the saved image the rendered image used for the uv map.
            Only used if save_render is True, defaults to 0.
        save_render : Bool
            Wether the textureless renders used for computing the UV mappings should be saved as images. Defaults to
            False.
        Returns
        -------
        warp
            Numpy array representing the UV mapping.
        """
        # TODO: not very efficient, consider using an element index array later
        # make vertex array from objloader.Obj object with two attributes, in_vert as vec3 and in_text as vec_2
        self.vao = self.ctx.simple_vertex_array(
            self.prog,
            self.ctx.buffer(self.object.pack('vx vy vz tx ty')),
            "in_vert", "in_text"
        )

        translation_matrix = Matrix44.from_translation((
            np.random.uniform(self.x_low, self.x_high),
            np.random.uniform(self.y_low, self.y_high),
            0.0
        ))

        rotation_matrix = Matrix44.from_matrix33(
            self.rand_rotation_matrix(self.deflection)
        )

        view_matrix = Matrix44.look_at(
            (0.0, 0.0, np.random.uniform(self.close, self.far)),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        projection_matrix = Matrix44.perspective_projection(
            45.0, self.width / self.height, 0.1, 1000.0
        )

        # TODO: translation or rotation first?
        transform = projection_matrix * view_matrix * translation_matrix * rotation_matrix

        # make frame buffer ready for new render
        self.fbo.use()
        self.fbo.clear()

        # use computed transformation matrix as the MVP which will be used in the vertex shader
        self.mvp.write(transform.astype('f4').tobytes())
        self.vao.render()

        if save_render:
            Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1).save(
                'training_renders/scene_{}.jpg'.format(i))

        framebuffer = self.fbo.read(components=2, dtype='f4')
        warp = np.frombuffer(framebuffer, dtype=np.float32).reshape(
            (self.height, self.width, 2))[::-1]

        return warp
