const TRANSFORMATIONS_GLSL = 
`
mat4 rotateZaxis(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
    vec4(c, -s, 0, 0),
    vec4(s, c, 0, 0),
    vec4(0, 0, 1, 0),
    vec4(0, 0, 0, 1)
    );
}

mat4 rotateYaxis(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
    vec4(c, 0, s, 0),
    vec4(0, 1, 0, 0),
    vec4(-s, 0, c, 0),
    vec4(0, 0, 0, 1)
    );
}

mat4 rotateXaxis(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
    vec4(1, 0, 0, 0),
    vec4(0, c, -s, 0),
    vec4(0,s, c, 0),
    vec4(0, 0, 0, 1)
    );
}

mat4 rotateXYZ(vec3 rotation) {
	return rotateXaxis(rotation.x) * rotateYaxis(rotation.y) * rotateZaxis(rotation.z);
}
`

export class Fractal {
	iterations = 6;
	colorIterations = 6;
	static DEFAULT_FRACTAL_COLOR = 'vec3(orbitTrap.x * 0.2, orbitTrap.y * 0.4, orbitTrap.z * 0.9) * orbitTrap.w + diffuse';
	getDistanceFunction = () => {
		
	}
}

export class MandelboxFractal extends Fractal {
	scale = 2.8;
	getDistanceFunction = () => {
		let glsl = `
		vec4 orbitTrap = vec4(MAX_DIST);
		float distanceFunction(vec3 pos, bool isLight, int reflectionIndex) {
			if(!isLight) orbitTrap = vec4(MAX_DIST);
			float scale = #SCALE;
			float MR2 = 0.2;
			vec4 scalevec = vec4(scale, scale, scale, abs(scale)) / MR2;
			float C1 = abs(scale - 1.0), C2 = pow(abs(scale), float(1 - #ITERATIONS));
			vec4 p = vec4(pos.xyz, 1.0), p0 = vec4(pos.xyz, 1.0);

			for (int i = 0; i < #ITERATIONS; i++) {
				p.xyz = clamp(p.xyz, -1.0, 1.0) * 2.0 - p.xyz;
				float r2 = dot(p.xyz, p.xyz);
				if (i < #COLOR_ITERATIONS && !isLight) 
					orbitTrap = min(orbitTrap, abs(vec4(p.xyz, r2)));
				p.xyzw *= clamp(max(MR2/r2, MR2), 0.0, 1.0);
				p.xyzw = p * scalevec + p0;
			}
			return ((length(p.xyz) - C1) / p.w) - C2;
		}
		`;
		glsl = glsl.replace(/#ITERATIONS/g, this.iterations);
		glsl = glsl.replace(/#COLOR_ITERATIONS/g, this.colorIterations);
		glsl = glsl.replace(/#SCALE/g, this.scale.toFixed(8));
		return glsl;
	}
}

export class SphereSpongeFractal extends Fractal {
	scale = 2.0;
	spongeScale = 2.05;
	getDistanceFunction = () => {
		let glsl = `
		vec4 orbitTrap = vec4(MAX_DIST);
		float distanceFunction(vec3 position, bool isLight, int reflectionIndex) {
			if(!isLight) orbitTrap = vec4(MAX_DIST);
			float scale = #SCALE;
			float spongeScale = #SPONGE_SCALE;
			float k = scale;
			float d = -MAX_DIST, md = MAX_DIST;
			float d1, r;

			for (int i = 0; i < #ITERATIONS; i++) {
				vec3 z = mod(position * k, 4.0) - vec3(0.5 * 4.0);
				r = length(z);
				d1 = (spongeScale - r) / k;
				k *= scale;
				d = max(d, d1);
				if (i < #COLOR_ITERATIONS && !isLight) {
					md = min(md, d);
					orbitTrap = vec4(md, md, md, r);
				}
			}
			return d;
		}
		`;
		glsl = glsl.replace(/#ITERATIONS/g, this.iterations);
		glsl = glsl.replace(/#COLOR_ITERATIONS/g, this.colorIterations);
		glsl = glsl.replace(/#SCALE/g, this.scale.toFixed(8));
		glsl = glsl.replace(/#SPONGE_SCALE/g, this.spongeScale.toFixed(8));
		return glsl;
	}
}

export class MandelbulbFractal extends Fractal {
	power = 10;
	bailout = 50;
	getDistanceFunction = () => {
		let glsl = `
		vec4 orbitTrap = vec4(MAX_DIST);
		float distanceFunction(vec3 position, bool isLight, int reflectionIndex) {
			position *= mat3(rotateXaxis(PI / 2.0));
			if(!isLight) orbitTrap = vec4(MAX_DIST);
			vec3 z = position;
			float dr = 1.0;
			float r = 0.0;
			for (int i = 0; i < #ITERATIONS; i++) {
				r = length(z);
				if (r > #BAILOUT) 
					break;
				float theta = acos(z.z/r);
				float phi = atan(z.y,z.x);
				dr =  pow( r, #POWER - 1.0 ) * #POWER * dr + 1.0;
				float zr = pow(r, #POWER);
				theta = theta * #POWER;
				phi = phi * #POWER;
				z = zr * vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
				z += position;
				if (i < #COLOR_ITERATIONS && !isLight) 
					orbitTrap = min(orbitTrap, abs(vec4(z.x, z.y, z.z, r*r)));
			}
			return 0.5*log(r)*r/dr;
		}
		`;
		glsl = glsl.replace(/#ITERATIONS/g, this.iterations);
		glsl = glsl.replace(/#COLOR_ITERATIONS/g, this.colorIterations);
		glsl = glsl.replace(/#POWER/g, this.power.toFixed(8));
		glsl = glsl.replace(/#BAILOUT/g, this.bailout.toFixed(8));
		return glsl;
	}
}

export class SierpinskiPyramidFractal extends Fractal {
	scale = 2.0;
	offset = 2.0;
	getDistanceFunction = () => {
		let glsl = `
		vec4 orbitTrap = vec4(MAX_DIST);
		float distanceFunction(vec3 position, bool isLight, int reflectionIndex) {
			const float scale = #SCALE;
			const float offset = #OFFSET;

			for(int i = 0; i < #ITERATIONS; i++) {
				position.xy = (position.x + position.y < 0.0) ? -position.yx : position.xy;
				position.xz = (position.x + position.z < 0.0) ? -position.zx : position.xz;
				position.zy = (position.z + position.y < 0.0) ? -position.yz : position.zy;
				position = scale * position - offset * (scale - 1.0);
				if(isLight && i < #COLOR_ITERATIONS) {
					orbitTrap = min(orbitTrap, abs(vec4(position.x, position.y, position.z, scale * offset)));
				}
			}
		 
			return length(position) * pow(scale, -float(#ITERATIONS));
		}
		`;
		glsl = glsl.replace(/#ITERATIONS/g, this.iterations);
		glsl = glsl.replace(/#COLOR_ITERATIONS/g, this.colorIterations);
		glsl = glsl.replace(/#SCALE/g, this.scale.toFixed(8));
		glsl = glsl.replace(/#OFFSET/g, this.offset.toFixed(8));
		return glsl;
	}
}

export class MengerSpongeFractal extends Fractal {
	scale = 1;
	offset = 1;
	getDistanceFunction = () => {
		let glsl = `
		float maxcomp(vec3 p) {
			float m1 = max(p.x, p.y);
			return max(m1, p.z);
		}

		vec2 objBoxS(vec3 p, vec3 b) {
			vec3  di = abs(p) - b;
			float mc = maxcomp(di);
			float d = min(mc, length(max(di, 0.0)));
			return vec2(d, 1);
		}

		vec2 objBox(vec3 p) {
			vec3 b = vec3(4.0);
			return objBoxS(p, b);
		}

		vec2 objCross(in vec3 p) {
			vec2 da = objBoxS(p.xyz, vec3(MAX_DIST, 2.0, 2.0));
			vec2 db = objBoxS(p.yzx, vec3(2.0, MAX_DIST, 2.0));
			vec2 dc = objBoxS(p.zxy, vec3(2.0, 2.0, MAX_DIST));
			return vec2(min(da.x, min(db.x, dc.x)), 1);
		}

		vec4 orbitTrap = vec4(MAX_DIST);
		float distanceFunction(vec3 position, bool isLight, int reflectionIndex) {
			vec2 d2 = objBox(position);
			float scale = #SCALE;
			float offset = #OFFSET;
			for(int i = 0; i < #ITERATIONS; i++) {
				vec3 a = mod(position * scale, 2.0) - offset;
				scale *= 3.0;
				vec3 r = 1.0 - 4.0 * abs(a);
				vec2 c = objCross(r) / scale;
				d2.x = max(d2.x, c.x);
				if(isLight && i < #COLOR_ITERATIONS) {
					orbitTrap = min(orbitTrap, abs(vec4(a.x, a.y, a.z, scale * offset)));
				}
			} 
			return d2.x;
		}
		`;
		glsl = glsl.replace(/#ITERATIONS/g, this.iterations);
		glsl = glsl.replace(/#COLOR_ITERATIONS/g, this.colorIterations);
		glsl = glsl.replace(/#SCALE/g, this.scale.toFixed(8));
		glsl = glsl.replace(/#OFFSET/g, this.offset.toFixed(8));
		return glsl;
	}
}

export class RayMarcher {
	vertexShaderCode = `
		precision highp float;
		attribute vec2 position;
		void main() {
			gl_Position = vec4(position, 0, 1);
		}
		`;
	fragmentShaderCode = `
		#define PI 3.1415925359
		#define MAX_STEPS #MAX_STEPS
		#define MAX_DIST #MAX_DISTANCE
		#define MIN_DIST #MIN_DISTANCE
		precision highp float;
		uniform vec2 uResolution;
		uniform float uTime;
		uniform vec3 uPosition;
		uniform vec3 uRotation;
		int steps = 0;
		${TRANSFORMATIONS_GLSL}

		#DISTANCE_FUNCTION
		
		float sceneDE(vec3 position, bool isLight, int reflectionIndex) {
			#SPIN
			return distanceFunction(position, isLight, reflectionIndex);
		}

		float minDistance = MIN_DIST;
		float lastDistance = 0.0;
		float rayMarch(vec3 rayPos, vec3 rayDir, bool isLight, int reflectionIndex) {
			float marchedDistance = 0.0;
			for(int i = 0; i < MAX_STEPS; i++) {
				steps = i;
				vec3 p = rayPos + rayDir * marchedDistance;
				float distance = sceneDE(p, isLight, reflectionIndex);
				if(#DYNAMIC_MIN_DIST && !isLight)
					minDistance = marchedDistance * MIN_DIST;
				marchedDistance += distance;
				if(marchedDistance > MAX_DIST || distance < minDistance) {
					lastDistance = distance;
					break;
				}
			}
			return marchedDistance;
		}

		vec3 getNormal(vec3 p, int reflectionIndex) {
			float distance = sceneDE(p, true, reflectionIndex);
			vec2 epsilon = vec2(minDistance, 0);
			vec3 n = distance - vec3(
				sceneDE(p - epsilon.xyy, true, reflectionIndex),
				sceneDE(p - epsilon.yxy, true, reflectionIndex),
				sceneDE(p - epsilon.yyx, true, reflectionIndex));
			return normalize(n);
		}

		float getLight(vec3 p, int reflectionIndex) { 
			vec3 lightPos = #LIGHT_FUNCTION;
			vec3 lightDir = normalize(lightPos-p);
			vec3 normal = getNormal(p, reflectionIndex);
			
			float diffuse = dot(normal, lightDir);
			diffuse = clamp(diffuse, 0.0, 1.0);
			
			#SHADOWS
		 
			return diffuse;
		}

		#EXTRA

		void main() {
			vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;
			
			vec3 cameraPos = uPosition;
			vec3 rayDir = normalize(vec3(uv.x, uv.y, #FOV));
			rayDir = rayDir * mat3(rotateXYZ(uRotation));
			
			vec3 fragColor;
			bool isFirstSky = false;
			
			for(int reflection = 0; reflection < #REFLECTIONS + 1; reflection++) {
				vec3 normal;
				if(reflection > 0 || #CALC_NORMAL) {
					normal = getNormal(cameraPos, reflection);
					if(reflection > 0) rayDir = reflect(rayDir, normal);
				}
				float distance = rayMarch(reflection > 0 ? cameraPos + normal * (2.0 * minDistance) : cameraPos, rayDir, false, reflection);
				cameraPos += rayDir * distance;
				float diffuse = getLight(cameraPos, reflection);
				vec3 color = #COLOR_FUNCTION;
				if(distance > MAX_DIST) {
					color = #SKY_COLOR_FUNCTION;
					if(reflection == 0) isFirstSky = true;
				}
				else if(steps == MAX_STEPS) {
					color = #SKY_COLOR_FUNCTION;
					if(reflection == 0) isFirstSky = true;
				}
				
				if(reflection > 0 && !isFirstSky) fragColor = mix(fragColor, color, #REFLECTNESS);
				else if(reflection > 0);
				else fragColor = color;
			}
			gl_FragColor = vec4(fragColor, 1.0);
		}
		`;
		
	maxDistance = 1000.0;
	maxSteps = 256;
	minDistance = 0.001;
	fov = 1.0;
	skyColorFunction = 'vec3(0, 0, 1)';
	colorFunction = 'vec3(diffuse)';
	lightFunction = 'vec3(0, -15, 0)';
	shadows = true;
	reflectness = 0.5;
	reflections = 0;
	extra = '';
	spin = false;
	dynamicMinDistance = true;
	calcNormal = false;
	shadowStrength = 0.9;
	
	pressed = new Set();

	toRadians = (deg) => {
		return deg * (Math.PI / 180);
	}

	keyPress = (keyEvent) => {
		this.pressed.add(keyEvent.code);
	}

	keyRelease = (keyEvent) => {
		this.pressed.delete(keyEvent.code);
	}

	getNewPosition = (currentPosition, currentRotation, speed) => {
		let newPosition = {...currentPosition};
		for(let key of this.pressed) {
			if(key == 'KeyA') {
				let dx = -speed * Math.sin(this.toRadians(currentRotation.y + 90));
				let dz = speed * Math.sin(this.toRadians(90 - currentRotation.y + 90));
				newPosition.x += dx;
				newPosition.z += dz;
			}
			if(key == 'KeyD') {
				let dx = speed * Math.sin(this.toRadians(currentRotation.y + 90));
				let dz = -speed * Math.sin(this.toRadians(90 - currentRotation.y + 90));
				newPosition.x += dx;
				newPosition.z += dz;
			}
			if(key == 'KeyW') {
				let dx = speed * Math.sin(this.toRadians(currentRotation.y));
				let dz = speed * Math.sin(this.toRadians(90 - currentRotation.y));
				newPosition.x += dx;
				newPosition.z += dz;
			}
			if(key == 'KeyS') {
				let dx = -speed * Math.sin(this.toRadians(currentRotation.y));
				let dz = -speed * Math.sin(this.toRadians(90 - currentRotation.y));
				newPosition.x += dx;
				newPosition.z += dz;
			}
			if(key == 'Space') newPosition.y += speed;
			if(key == 'ShiftLeft') newPosition.y -= speed;
		}
		return newPosition;
	}

	prevMouseX = 0;
	prevMouseY = 0;

	getNewRotation = (currentRotation, mouseMoveEvent, isMousePressed, speed) => {
		let newRotation = {...currentRotation};
		if(isMousePressed) {
			let dxDeg = (mouseMoveEvent.offsetX - this.prevMouseX) * speed;
			let dyDeg = (mouseMoveEvent.offsetY - this.prevMouseY) * speed;
			newRotation.x += dyDeg;
			newRotation.y += dxDeg;
		}
		this.prevMouseX = mouseMoveEvent.offsetX;
		this.prevMouseY = mouseMoveEvent.offsetY;
		return newRotation;
	}

	gl;
	canvas;
	uPosition;
	uTime;
	uResolution;
	uRotation;
	cameraPosition = {x: 0, y: 0, z: 0};
	cameraRotation = {x: 0, y: 0, z: 0};
	mouseMoveEvent;
	isMousePressed;
	moveSpeed = 0.01;
	lookSpeed = 0.25;
	frameIntervalMS = 1;
	moveOnlyWhenMouseInside = true;
	mouseInside = false;
	moving = false;
	previewScale = 4;
	mouseWheelFactor = 1.5;
	pauseWhenNotMoving = true;
	canvasStartDimention = {};
	doEveryFrame = function(){};
	disableInput = false;

	moveStartTime;
	setMoving = () => {
		this.moving = true;
		this.moveStartTime = this.totalTime;
	}

	totalTime = 0;
	update = (delta) => {
		if(this.isMousePressed) this.setMoving();
		if(this.moving) {
			this.canvas.width = this.canvasStartDimention.x / this.previewScale;
			this.canvas.height = this.canvasStartDimention.y / this.previewScale;
		}
		if(this.mouseMoveEvent) this.cameraRotation = this.getNewRotation(this.cameraRotation, this.mouseMoveEvent, this.isMousePressed, this.lookSpeed * (window.innerWidth / this.canvasStartDimention.x));
		if(!this.moving && !this.spin && this.pauseWhenNotMoving) return;
		if(this.totalTime - this.moveStartTime > 1 && !this.spin && this.pauseWhenNotMoving) {
			this.moving = false;
			this.moveStartTime = 0;
			this.canvas.width = this.canvasStartDimention.x;
			this.canvas.height = this.canvasStartDimention.y;
		}
		this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
		this.cameraPosition = this.getNewPosition(this.cameraPosition, this.cameraRotation, this.moveSpeed);
		this.totalTime += delta;
		this.gl.uniform2f(this.uResolution, this.canvas.width, this.canvas.height);
		this.gl.uniform1f(this.uTime, this.totalTime);
		this.gl.uniform3f(this.uPosition, this.cameraPosition.x, this.cameraPosition.y, this.cameraPosition.z);
		this.gl.uniform3f(this.uRotation, this.toRadians(this.cameraRotation.x % 360), this.toRadians(this.cameraRotation.y % 360), this.toRadians(this.cameraRotation.z % 360));
		this.doEveryFrame();
		this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
		this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
	}
	
	compile = (canvas, distanceFunction) => {
		this.setMoving();
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#DISTANCE_FUNCTION/g, distanceFunction);
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#COLOR_FUNCTION/g, this.colorFunction);
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#SKY_COLOR_FUNCTION/g, this.skyColorFunction);
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#MAX_DISTANCE/g, this.maxDistance.toFixed(8));
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#MIN_DISTANCE/g, this.minDistance.toFixed(8));
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#MAX_STEPS/g, this.maxSteps);
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#FOV/g, this.fov);
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#LIGHT_FUNCTION/g, this.lightFunction);
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#REFLECTNESS/g, this.reflectness.toFixed(8));
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#REFLECTIONS/g, this.reflections);
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#EXTRA/g, this.extra);
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#DYNAMIC_MIN_DIST/g, this.dynamicMinDistance);
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#CALC_NORMAL/g, this.calcNormal);
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#SPIN/g, 
				this.spin ? 'position *= mat3(rotateYaxis(mod(uTime / 2.0, 2.0 * PI)));' : '');
		this.fragmentShaderCode = this.fragmentShaderCode.replace(/#SHADOWS/g, 
							!this.shadows ? '' : `
							float d = rayMarch(p + normal * minDistance * 2.0, lightDir, true, reflectionIndex);
							if(d < length(lightPos-p)) diffuse *= ${1 - this.shadowStrength};
							`);
							
		this.canvas = canvas;
		this.gl = canvas.getContext('webgl');
		if(!this.gl)
			this.gl = canvas.getContext('experimental-webgl');
		
		this.canvasStartDimention.x = canvas.width;
		this.canvasStartDimention.y = canvas.height;

		let vertexShader = this.gl.createShader(this.gl.VERTEX_SHADER);
		let fragmentShader = this.gl.createShader(this.gl.FRAGMENT_SHADER);
			
		this.gl.shaderSource(vertexShader, this.vertexShaderCode);
		this.gl.shaderSource(fragmentShader, this.fragmentShaderCode);
			
		this.gl.compileShader(vertexShader);
		if(!this.gl.getShaderParameter(vertexShader, this.gl.COMPILE_STATUS))
			throw ("compilation error: " + this.gl.getShaderInfoLog(vertexShader));
		this.gl.compileShader(fragmentShader);
		if(!this.gl.getShaderParameter(fragmentShader, this.gl.COMPILE_STATUS))
			throw ("compilation error: " + this.gl.getShaderInfoLog(fragmentShader));
			
		let program = this.gl.createProgram();
		this.gl.attachShader(program, vertexShader);
		this.gl.attachShader(program, fragmentShader);
			
		this.gl.linkProgram(program);
		if(!this.gl.getProgramParameter(program, this.gl.LINK_STATUS))
			throw ("linking error: " + this.gl.getProgramInfoLog(program));
			
		this.gl.validateProgram(program);
		if(!this.gl.getProgramParameter(program, this.gl.VALIDATE_STATUS))
			throw ("validation error: " + this.gl.getProgramInfoLog(program));
		
		this.uTime = this.gl.getUniformLocation(program, 'uTime');
		this.uResolution = this.gl.getUniformLocation(program, 'uResolution');
		this.uPosition = this.gl.getUniformLocation(program, 'uPosition');
		this.uRotation = this.gl.getUniformLocation(program, 'uRotation');
		
		let vertices = new Float32Array([-1, 1, -1, -1, 1, 1, 1, -1]);
		let vao = this.gl.createBuffer();
		this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vao);
		this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);
		this.gl.vertexAttribPointer(0, 2, this.gl.FLOAT, this.gl.FALSE, 2 * Float32Array.BYTES_PER_ELEMENT, 0);
		this.gl.enableVertexAttribArray(0);

		this.gl.useProgram(program);
		
		if(this.frameIntervalMS != -1)
			window.setInterval(() => this.update(this.frameIntervalMS / 100), this.frameIntervalMS);
		
		if(!this.disableInput) {
			window.addEventListener('keydown', e => {
				if(this.moveOnlyWhenMouseInside && !this.mouseInside) return;
				this.setMoving();
				this.keyPress(e);
			});
			
			window.addEventListener('keyup', e => this.keyRelease(e));
			window.addEventListener('mousemove', e => this.mouseMoveEvent = e);
			canvas.addEventListener('mouseup', e => this.isMousePressed = false);
			canvas.addEventListener('mousedown', e => this.isMousePressed = true);
			canvas.onmouseenter = () => this.mouseInside = true;
			canvas.onmouseout = () => this.mouseInside = false;
			
			canvas.addEventListener('mousewheel', e => {
				let delta = e.deltaY / 40.0;
				if(delta < 0) this.moveSpeed *= this.mouseWheelFactor;
				else this.moveSpeed /= this.mouseWheelFactor;
			});
		}
		
		if(this.frameIntervalMS == -1) this.update(0);
		canvas.style.width = this.canvasStartDimention.x;
		canvas.style.height = this.canvasStartDimention.y;
	}
}