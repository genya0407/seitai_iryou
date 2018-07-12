extern crate nalgebra as na;
extern crate rand;
use std::fmt;
use rand::prelude::*;

fn original_image() -> Image {
    let mut original_image = na::DMatrix::<f32>::zeros(5, 5);
    original_image[(1, 2)] = 10.0;
    original_image[(2, 1)] = 10.0;
    original_image[(2, 2)] = 10.0;
    original_image[(2, 3)] = 10.0;
    original_image[(3, 1)] = 10.0;
    original_image[(3, 2)] = 10.0;
    original_image[(3, 3)] = 10.0;
    return Image::new(original_image);
}

fn random_image(nrows: usize, ncols: usize) -> Image {
    let mut rng = thread_rng();
    let data = na::DMatrix::<f32>::from_fn(nrows, ncols, |_, _| rng.gen_range(0, 100) as f32);
    Image::new(data)
}

fn sparse_random_image(nrows: usize, ncols: usize) -> Image {
    let mut random_image = random_image(nrows, ncols);
    random_image.data.apply(|val| {
        if val > 70. {
            val
        } else {
            0.
        }
    });
    random_image
}

#[derive(Clone)]
struct Image {
    pub data: na::DMatrix<f32>
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl Image {
    fn new(data: na::DMatrix<f32>) -> Self {
        Self { data: data }
    }

    fn rows_projection(&self) -> na::DVector<f32> {
        let mut v = na::DVector::<f32>::zeros(self.data.ncols());
        v.fill(1.);
        self.data.clone() * v
    }

    fn cols_projection(&self) -> na::DVector<f32> {
        let mut v = na::DVector::<f32>::zeros(self.data.nrows());
        v.fill(1.);
        self.data.clone().transpose() * v
    }

    fn round(&self) -> Self {
        let mut img = self.clone();
        img.data.apply(|val| val.round());
        return img;
    }
}

struct ImageInferrer {
    pub image: Image,
    pub rows_proj: na::DVector<f32>,
    pub cols_proj: na::DVector<f32>,
}

impl ImageInferrer {
    fn infer_by_back_projection(rows_proj: na::DVector<f32>, cols_proj: na::DVector<f32>) -> Self {
        let mut data = na::DMatrix::<f32>::zeros(rows_proj.len(), cols_proj.len());

        for (ri, proj_val) in rows_proj.iter().enumerate() {
            let mean_val = proj_val / rows_proj.len() as f32;
            for ci in 0..cols_proj.len() {
                data[(ri, ci)] += mean_val;
            }
        }
        for (ci, proj_val) in cols_proj.iter().enumerate() {
            let mean_val = proj_val / cols_proj.len() as f32;
            for ri in 0..rows_proj.len() {
                data[(ri, ci)] += mean_val;
            }
        }

        let image = Image::new(data);
        Self { image: image, rows_proj: rows_proj, cols_proj: cols_proj }
    }

    fn adjust(&mut self) {
        for (ri, (computed_val, target_val)) in self.image.rows_projection().iter().zip(self.rows_proj.iter()).enumerate() {
            let diff = target_val - computed_val;
            let mean_val = diff / self.image.data.ncols() as f32;
            for ci in 0..self.image.data.ncols() {
                self.image.data[(ri, ci)] = 0.0_f32.max(self.image.data[(ri, ci)] + mean_val);
            }
        }
        for (ci, (computed_val, target_val)) in self.image.cols_projection().iter().zip(self.cols_proj.iter()).enumerate() {
            let diff = target_val - computed_val;
            let mean_val = diff / self.image.data.nrows() as f32;
            for ri in 0..self.image.data.nrows() {
                self.image.data[(ri, ci)] = 0.0_f32.max(self.image.data[(ri, ci)] + mean_val);
            }
        }
    }
}

fn main() {
    println!("===== Given Image =====\n");
    let orig_image = original_image();
    println!("Original Image:");
    println!("{}", orig_image);
    let rows_proj = orig_image.rows_projection();
    let cols_proj = orig_image.cols_projection();
    let mut inferer = ImageInferrer::infer_by_back_projection(rows_proj, cols_proj);
    println!("Back Projection:");
    println!("{}", inferer.image.round());
    for _ in 0..100 {
        inferer.adjust();
    }
    println!("Improved Back Projection:");
    println!("{}", inferer.image.round());

    println!("===== Random Image =====\n");
    let orig_image = sparse_random_image(5, 5);
    println!("Original Image:");
    println!("{}", orig_image);
    let rows_proj = orig_image.rows_projection();
    let cols_proj = orig_image.cols_projection();
    let mut inferer = ImageInferrer::infer_by_back_projection(rows_proj, cols_proj);
    println!("Back Projection:");
    println!("{}", inferer.image.round());
    for _ in 0..100 {
        inferer.adjust();
    }
    println!("Improved Back Projection:");
    println!("{}", inferer.image.round());
}
