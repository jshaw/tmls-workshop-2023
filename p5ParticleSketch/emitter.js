// The particle emitter and image texture was referenced and inspired by The Coding Train's Particle system projects
// The Nature of Code
// The Coding Train / Daniel Shiffman / Dusk
// 
// https://youtu.be/pUhv2CA0omA
// https://thecodingtrain.com
// Particles
// https://editor.p5js.org/codingtrain/sketches/TTVoNt58T

class Emitter {
    constructor(x, y, img, offset) {
      this.position = createVector(x, y);
      this.particles = [];
      this.img = img;
      this.offset = offset
    }
  
    emit(num) {
      for (let i = 0; i < num; i++) {
        this.particles.push(new Particle(this.position.x, this.position.y, this.img));
      }
    }
    
    updatePos(x,y, _offset){
      if(_offset){
        this.position = createVector(x+this.offset, y+this.offset);
      } else {
        this.position = createVector(x, y);
      }
    }
  
    applyForce(force) {
      for (let particle of this.particles) {
        particle.applyForce(force);
      }
    }
  
    update() {
      for (let particle of this.particles) {
        particle.update();
      }
  
      for (let i = this.particles.length - 1; i >= 0; i--) {
        if (this.particles[i].finished()) {
          this.particles.splice(i, 1);
        }
      }
    }
  
    show(particleData) {
      let loopMax = this.particles.length;
      if (loopMax > numParticles) {
        loopMax = numParticles;
      }
      for (let i = 0; i < loopMax; i++) {
        particleData[i].set(
          map(this.particles[i].pos.x, 0, width, 0, 1),
          map(this.particles[i].pos.y, 0, height, 0, 1),
          map(this.particles[i].lifetime, 0, 255, 0, 1)
        );
      }
      
      for (let particle of this.particles) {
        particle.show();
      }
    }
  }
  