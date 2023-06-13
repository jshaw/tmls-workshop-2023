// The particle emitter and image texture was referenced and inspired by The Coding Train's Particle system projects
// The Nature of Code
// The Coding Train / Daniel Shiffman / Dusk
// 
// https://youtu.be/pUhv2CA0omA
// https://thecodingtrain.com
// Particles
// https://editor.p5js.org/codingtrain/sketches/TTVoNt58T

class Particle {
    constructor(x, y, img) {
      this.pos = createVector(x, y);
      this.vel = p5.Vector.random2D();
      this.vel.mult(random(0.5, 2));
      this.acc = createVector(0, 0);
      this.r = 64;
      this.lifetime = 255;
      this.img = img;
    }
  
    finished() {
      return this.lifetime < 0;
    }
  
    applyForce(force) {
      this.acc.add(force);
    }
  
    update() {
      this.vel.add(this.acc);
      this.pos.add(this.vel);
      this.acc.set(0, 0);
  
      this.lifetime -= 5;
    }
  
    show() {
      imageMode(CENTER);
      image(img, this.pos.x, this.pos.y, this.r, this.r);
    }
  }
  