extern crate nalgebra as na;
use std::fmt;

// 逆投影をした後に反復的に改善を繰り返すことで、より鮮明な像を得ることができるのではないか？
// 改善：列、行の合計とスキャン値を比較し、合計値がスキャン値に近づくように修正を加える。

struct Image {
    pub data: na::DMatrix<i32>
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl Image {
    fn new(data: na::DMatrix<i32>) -> Self {
        Self { data: data }
    }

    fn rows_projection(&self) -> na::DVector<i32> {
        let mut v = na::DVector::<i32>::zeros(self.data.nrows());
        v.fill(1);
        self.data.clone() * v
    }

    fn cols_projection(&self) -> na::DVector<i32> {
        let mut v = na::DVector::<i32>::zeros(self.data.nrows());
        v.fill(1);
        self.data.clone().transpose() * v
    }
}

fn original_image() -> Image {
    let mut original_image = na::DMatrix::<i32>::zeros(5, 5);
    original_image[(1, 2)] = 10;
    original_image[(2, 1)] = 10;
    original_image[(2, 2)] = 10;
    original_image[(2, 3)] = 10;
    original_image[(3, 1)] = 10;
    original_image[(3, 2)] = 10;
    original_image[(3, 3)] = 10;
    return Image::new(original_image);
}

fn main() {
    let orig_image = original_image();
    println!("{}", orig_image);
    println!("{}", orig_image.rows_projection());
    println!("{}", orig_image.cols_projection());
}
