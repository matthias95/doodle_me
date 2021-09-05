#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <memory>

template<class VT>
class Matrix {
private:
    
    
    const bool do_free_buffer=false;
    

public:
    const size_t row_off=0;
    const size_t col_off=0;
    const size_t height;
    const size_t width; 
    const size_t height_buffer;
    const size_t width_buffer; 
    const size_t channels;
    VT* const buffer;

    Matrix(const size_t height, const size_t width, const size_t channels=1, bool do_free_buffer=true):
    height{height}, 
    width{width},
    height_buffer(height),
    width_buffer(width),
    channels{channels}, 
    buffer{(VT*)malloc(height * width * channels * sizeof(VT))},
    do_free_buffer{do_free_buffer}
    {
    };

    Matrix(VT* const buffer , const size_t height, const size_t width, const size_t channels=1,  const size_t row_off=0, const size_t col_off=0, const size_t height_buffer=0, const size_t width_buffer=0):
    height{height}, 
    width{width}, 
    height_buffer{height_buffer == 0 ? height : height_buffer},
    width_buffer{width_buffer == 0 ? width : width_buffer},
    channels{channels}, 
    buffer{buffer},
    row_off{row_off},
    col_off{col_off},
    do_free_buffer{false}
    {
    };

    Matrix():
    height{0}, 
    width{0}, 
    height_buffer{0},
    width_buffer{0},
    channels{0}, 
    buffer{nullptr},
    row_off{0},
    col_off{0},
    do_free_buffer{false}
    {
    };


    ~Matrix(){
        if(do_free_buffer){
            free(buffer);
        }
    }

    VT& operator()(const size_t idx) {
        const size_t row = idx / (width * channels);
        const size_t col = idx % (width * channels);
        return buffer[(row + row_off) * width_buffer * channels + (col + col_off * channels)];
    }


    VT& operator()(const size_t row, const size_t col, const size_t ch=0) {
        return buffer[(row + row_off) * width_buffer * channels + (col + col_off) * channels + ch];
    }

    const VT& operator()(const size_t row, const size_t col, const size_t ch=0) const {
        return buffer[(row + row_off) * width_buffer * channels + (col + col_off) * channels + ch];
    }

    Matrix<VT> operator()(int row_min, int row_max, int col_min, int col_max) {
        if(row_max < 0){
            row_max += height;
        }
        if(col_max < 0){
            col_max += width;
        }
        return Matrix(buffer, row_max-row_min, col_max-col_min, channels, row_off + row_min, col_off+col_min, height_buffer, width_buffer);
    }

};

constexpr float PI = 3.141592653589793;
constexpr float E = 2.718281828459045;

Matrix<float> get_gaus_mat(int k_size, float std_dev){
    Matrix<float> gaus_mat(k_size, k_size,1,false);
    int center = (k_size / 2);
    float sum = 0;
    for(int row = 0; row < k_size; ++row){
        for(int col = 0; col < k_size; ++col){
            const int y = row - center;
            const int x = col - center;
            const float var = std_dev*std_dev;
            gaus_mat(row,col) = (1/(2*PI*var)) * std::pow(E, -(x*x+y*y)/(2*var));
            sum += gaus_mat(row,col);
        }
    }
    for(int row = 0; row < k_size; ++row){
        for(int col = 0; col < k_size; ++col){
            gaus_mat(row,col) /= sum;
        }
    }
    return gaus_mat;
}


Matrix<float> get_sobel_mat_x(int k_size){
    Matrix<float> sobel_mat(k_size, k_size,1,false);
    int center = (k_size / 2);
    for(int row = 0; row < k_size; ++row){
        for(int col = 0; col < k_size; ++col){
            const int y = row - center;
            const int x = col - center;
            if(x!=0 || y!=0){
                sobel_mat(row,col) = x / static_cast<float>(y*y + x*x);
            } else {
                sobel_mat(row,col) = 0;
            }
        }
    }
    return sobel_mat;
}

Matrix<float> get_sobel_mat_y(int k_size){
    Matrix<float> sobel_mat(k_size, k_size,1,false);
    int center = (k_size / 2);
    for(int row = 0; row < k_size; ++row){
        for(int col = 0; col < k_size; ++col){
            const int y = row - center;
            const int x = col - center;
            if(x!=0 || y!=0){
                sobel_mat(row,col) = y / static_cast<float>(y*y + x*x);
            } else {
                sobel_mat(row,col) = 0;
            }
        }
    }
    return sobel_mat;
}

const Matrix<float>& get_gaus_weights(int k_size, float std_dev) {
    static std::unique_ptr<const Matrix<float>> gaus_mats[39][39];
    auto& w = gaus_mats[k_size-1][int(std_dev*10)-1];
    if(w == nullptr){
         w = std::make_unique<Matrix<float>>( get_gaus_mat(k_size, std_dev));
    }
    return *w;
}

enum SOBEL_DIR{
    SOBEL_DIR_X,
    SOBEL_DIR_Y
};

const Matrix<float>& get_sobel_weights(int k_size, SOBEL_DIR dir) {
    static std::unique_ptr<const Matrix<float>> gaus_mats[20][2];
    auto& w = gaus_mats[k_size-1][dir];
    if(w == nullptr){
        if(dir == SOBEL_DIR_X){
            w = std::make_unique<Matrix<float>>(get_sobel_mat_x(k_size));
        } else {
            w = std::make_unique<Matrix<float>>(get_sobel_mat_y(k_size));
        }
    }
    return *w;
}

template<class VT>
int draw_circle_bound_check(Matrix<VT>& img, int center_y, int center_x,  int r, VT color) {
    for(int y = -r; y < r+1; ++y) {
        if (((y+center_y) < 0) || ((y+center_y)>img.height)) {
            continue;
        }
        int x_width = std::sqrt(r*r - y*y);
        
        for(int x = center_x-x_width; x < center_x+1+x_width; ++x) {
            if (((center_x-x_width) < 0) || ((center_x+1+x_width) > img.width)) {
                continue;
            }
            img(y+center_y,x) = color;
        }
    }
    return 0;
}

template<class VT>
int draw_circle_no_bound_check(Matrix<VT>& img, int center_y, int center_x,  int r, VT color) {
    for(int y = -r; y < r+1; ++y){
        int x_width = std::sqrt(r*r - y*y);
        for(int x = (center_x-x_width); x < (center_x+1+x_width); ++x) {
            img(y+center_y,x) = color;
        }
    }
    return 0;
}

template<class VT>
int draw_circle(Matrix<VT>& img, int center_y, int center_x, int r, VT color) {
    const int height = img.height;
    const int width = img.width;
    if((center_y - r) < 0 || (center_y - r) >= height || (center_y + r) < 0 || (center_y + r) >= height || (center_x - r) < 0 || (center_x - r) >= width || (center_x + r) < 0 || (center_x + r) >= width){
        return draw_circle_bound_check(img, center_y, center_x, r, color);
    }

    return draw_circle_no_bound_check(img, center_y, center_x, r, color);
}

template<class VT>
int draw_circle(VT* img_buffer, int height, int width, int center_y, int center_x, int r, VT color) {
    Matrix<VT> img(img_buffer, height, width);
    return draw_circle( img, center_y, center_x, r, color);
}

template<class VT>
int draw_circle(Matrix<VT>& img, std::tuple<int,int> center, int r, VT color) {
    return draw_circle(img, std::get<0>(center), std::get<1>(center), r, color);
}

template<class VT>
int draw_line(Matrix<VT>& img, const int p1[2], const int p2[2], VT color=1, int width=1){
    const int p21[]  = {p2[0] - p1[0], p2[1] - p1[1]};
    const double length = sqrt(p21[0]*p21[0] + p21[1]*p21[1]);

    const double delta[] =  {p21[0] / length, p21[1] / length};

    for(int i = 0; i < ceil(length); ++i) {
        for(int w = -(width/2); w < (width/2 + 0.5); ++w){
            const int y = (p1[0] + w + delta[0] * i) + 0.5;
            const int x = (p1[1] + w + delta[1] * i) + 0.5;
            if(y >= 0 && y < img.height && x >= 0 && x < img.width)
                img(y,x, 0) = color;
        }
    }

    return 0;
}

template<class VT>
int draw_line(VT*  buffer, const size_t height, const size_t width, const int p1[2], const int p2[2], VT color=1){
    Matrix<VT> img(buffer, height, width);
    return draw_line(img, p1, p2, color);
}

template<class VT>
int draw_line_polar(Matrix<VT>& img, const int p[2], const float angle, const float mag, VT color=1){
        const int y = p[0];
        const int x = p[1];

        const int x_ = round(mag * cos(angle));
        const int y_ = round(mag * sin(angle));

        const int p1[] = {y-y_, x-x_};
        const int p2[] = {y+y_, x+x_};

        draw_line(img, p1, p2, color, 1);

    return 0;
}

template<class VT>
int draw_line_polar(VT*  buffer, const size_t height, const size_t width, const int p[2], const float angle, const float mag, VT color=1){
    Matrix<VT> img(buffer, height, width);
    return draw_line_polar(img, p, angle, mag, color);
}

template<class VT>
std::tuple<int,int> argmin(Matrix<VT>& img){
    VT min_val = std::numeric_limits<VT>::max();
    std::tuple<int,int> max_coords;

    for(size_t row = 0; row < img.height; ++row){
        for(size_t col = 0; col < img.width; ++col){
            const VT tmp = img(row,col,0);
            if(tmp < min_val){
                max_coords = std::make_tuple(row,col);
                min_val = tmp;
            }
        }
    }

    return max_coords;
}

template<class VT>
int argmin(VT*  buffer, const size_t height, const size_t width, int res[2]){
    Matrix<VT> img(buffer, height, width);
    auto tmp = argmin(img);
    res[0] = std::get<0>(tmp);
    res[1] = std::get<1>(tmp);
    return 0;
}

template<class VT>
int blur(Matrix<VT>& img, const int kernel_size, Matrix<VT>& dst){
    for(int row = 0; row < img.height; ++row){
        for(int col = 0; col < img.width; ++col){
            int k_row_start = row - (kernel_size/2);
            if(k_row_start < 0){
                k_row_start = 0;
            }

            int k_row_end = row + (int)(kernel_size/2.0 + 0.5);
            if(k_row_end > img.height){
                k_row_end = img.height;
            }

            int k_col_start = col - (kernel_size/2);
            if(k_col_start < 0){
                k_col_start = 0;
            }

            int k_col_end = col + (int)(kernel_size/2.0 + 0.5);
            if(k_col_end > img.width){
                k_col_end = img.width;
            }

            int cnt = 0;
            VT sum = 0;
            for(int k_row = k_row_start; k_row < k_row_end; ++k_row) {
                for(int k_col = k_col_start; k_col < k_col_end; ++k_col) {
                    sum += img(k_row, k_col);
                    cnt += 1;
                }
            }

            if(cnt > 0){
                dst(row,col) = sum / cnt;
            } else {
                dst(row,col) = 1;
            }
        }
    }
    return 0;
}

template<class VT>
int erode(Matrix<VT>& img, const int kernel_size, Matrix<VT>& dst){
    float weights[3][3] = {
        {0,1,0},
        {1,1,1},
        {0,1,0}
    };

    for(int row = 0; row < img.height; ++row){
        for(int col = 0; col < img.width; ++col){
            int k_row_start = row - (kernel_size/2);
            if(k_row_start < 0){
                k_row_start = 0;
            }

            int k_row_end = row + (int)(kernel_size/2.0 + 0.5);
            if(k_row_end > img.height){
                k_row_end = img.height;
            }

            int k_col_start = col - (kernel_size/2);
            if(k_col_start < 0){
                k_col_start = 0;
            }

            int k_col_end = col + (int)(kernel_size/2.0 + 0.5);
            if(k_col_end > img.width){
                k_col_end = img.width;
            }

            VT min_val = std::numeric_limits<VT>::max();
            for(int k_row = k_row_start; k_row < k_row_end; ++k_row) {
                for(int k_col = k_col_start; k_col < k_col_end; ++k_col) {
                    const int y = (k_row - row) + (kernel_size / 2);
                    const int x = (k_col - col) + (kernel_size / 2);
                    if(kernel_size != 3 || weights[y][x] > 0)
                        min_val = std::min(img(k_row, k_col), min_val);
                }
            }
            dst(row,col) = min_val;
        }
    }
    return 0;
}

template<class VT>
int dilate(Matrix<VT>& img, const int kernel_size, Matrix<VT>& dst){
    for(int row = 0; row < img.height; ++row){
        for(int col = 0; col < img.width; ++col){
            int k_row_start = row - (kernel_size/2);
            if(k_row_start < 0){
                k_row_start = 0;
            }

            int k_row_end = row + (int)(kernel_size/2.0 + 0.5);
            if(k_row_end > img.height){
                k_row_end = img.height;
            }

            int k_col_start = col - (kernel_size/2);
            if(k_col_start < 0){
                k_col_start = 0;
            }

            int k_col_end = col + (int)(kernel_size/2.0 + 0.5);
            if(k_col_end > img.width){
                k_col_end = img.width;
            }

            VT max_val = std::numeric_limits<VT>::min();
            for(int k_row = k_row_start; k_row < k_row_end; ++k_row) {
                for(int k_col = k_col_start; k_col < k_col_end; ++k_col) {
                    max_val = std::max(img(k_row, k_col), max_val);
                }
            }
            dst(row,col) = max_val;
        }
    }
    return 0;
}

template<class VT>
int blur(VT* img, const size_t height, const size_t width, const int kernel_size, VT* dst) {
    Matrix<VT> img_mat(img, height, width);
    Matrix<VT> dst_mat(dst, height, width);
    return blur(img_mat, kernel_size, dst_mat);
}

#ifdef Aasasas
template<class VT>
int gaus_blur(Matrix<VT>& img, const int kernel_size, const float std_dev, Matrix<VT>& dst){
    const Matrix<float>& weights = get_gaus_weights(kernel_size, std_dev);

    for(int row = 0; row < img.height; ++row){
        for(int col = 0; col < img.width; ++col){
            int k_row_start = row - (kernel_size/2);
            if(k_row_start < 0){
                k_row_start = 0;
            }

            int k_row_end = row + (int)(kernel_size/2.0 + 0.5);
            if(k_row_end > img.height){
                k_row_end = img.height;
            }

            int k_col_start = col - (kernel_size/2);
            if(k_col_start < 0){
                k_col_start = 0;
            }

            int k_col_end = col + (int)(kernel_size/2.0 + 0.5);
            if(k_col_end > img.width){
                k_col_end = img.width;
            }

            float cnt = 0;
            VT sum = 0;
            for(int k_row = k_row_start; k_row < k_row_end; ++k_row) {
                for(int k_col = k_col_start; k_col < k_col_end; ++k_col) {
                    
                    const int y = (k_row - row) + (kernel_size / 2);
                    const int x = (k_col - col) + (kernel_size / 2);
                    sum += img(k_row, k_col) * weights(y,x);
                    cnt += weights(y,x);
                }
            }

            if(cnt > 0){
                dst(row,col) = sum / cnt;
            } else {
                dst(row,col) = 1;
            }
        }
    }
    return 0;
}
#else
template<class VT>
int gaus_blur(Matrix<VT>& img, const int kernel_size, const float std_dev, Matrix<VT>& dst){
    const Matrix<float>& weights = get_gaus_weights(kernel_size, std_dev);
    Matrix<VT> dst_2(dst.height, dst.width, dst.channels);

    for(int row = 0; row < img.height; ++row){
        for(int col = 0; col < img.width; ++col){
            int k_row_start = row - (kernel_size/2);
            if(k_row_start < 0){
                k_row_start = 0;
            }

            int k_row_end = row + (int)(kernel_size/2.0 + 0.5);
            if(k_row_end > img.height){
                k_row_end = img.height;
            }

            int k_col_start = col - (kernel_size/2);
            if(k_col_start < 0){
                k_col_start = 0;
            }

            int k_col_end = col + (int)(kernel_size/2.0 + 0.5);
            if(k_col_end > img.width){
                k_col_end = img.width;
            }

            float cnt = 0;
            VT sum = 0;
            int k_col = col;
            for(int k_row = k_row_start; k_row < k_row_end; ++k_row) {
                const int y = (k_row - row) + (kernel_size / 2);
                sum += img(k_row, k_col) * weights(0,y);
                cnt += weights(0,y);

            }

            if(cnt > 0){
                dst_2(row,col) = sum / cnt;
            } else {
                dst_2(row,col) = 1;
            }
        }
    }

    for(int row = 0; row < img.height; ++row){
        for(int col = 0; col < img.width; ++col){
            int k_row_start = row - (kernel_size/2);
            if(k_row_start < 0){
                k_row_start = 0;
            }

            int k_row_end = row + (int)(kernel_size/2.0 + 0.5);
            if(k_row_end > img.height){
                k_row_end = img.height;
            }

            int k_col_start = col - (kernel_size/2);
            if(k_col_start < 0){
                k_col_start = 0;
            }

            int k_col_end = col + (int)(kernel_size/2.0 + 0.5);
            if(k_col_end > img.width){
                k_col_end = img.width;
            }

            float cnt = 0;
            VT sum = 0;
            int k_row = row;
            for(int k_col = k_col_start; k_col < k_col_end; ++k_col) {
                const int x = (k_col - col) + (kernel_size / 2);
                sum += dst_2(k_row, k_col) * weights(0,x);
                cnt += weights(0,x);
            }
            
            if(cnt > 0){
                dst(row,col) = sum / cnt;
            } else {
                dst(row,col) = 1;
            }
        }
    }
    return 0;
}
#endif

template<class VT>
int gaus_blur(VT* img, const size_t height, const size_t width, const int kernel_size, const float std_dev, VT* dst) {
    Matrix<VT> img_mat(img, height, width);
    Matrix<VT> dst_mat(dst, height, width);
    return gaus_blur(img_mat, kernel_size, std_dev, dst_mat);
}

template<class VT>
int sobel_filter(const Matrix<VT>& img, const int kernel_size, const SOBEL_DIR dir, Matrix<VT>& dst){
    const Matrix<float>& weights = get_sobel_weights(kernel_size, dir);

    for(int row = 0; row < img.height; ++row){
        for(int col = 0; col < img.width; ++col){
            int k_row_start = row - (kernel_size/2);
            if(k_row_start < 0){
                k_row_start = 0;
            }

            int k_row_end = row + (int)(kernel_size/2.0 + 0.5);
            if(k_row_end > img.height){
                k_row_end = img.height;
            }

            int k_col_start = col - (kernel_size/2);
            if(k_col_start < 0){
                k_col_start = 0;
            }

            int k_col_end = col + (int)(kernel_size/2.0 + 0.5);
            if(k_col_end > img.width){
                k_col_end = img.width;
            }

            VT sum = 0;
            for(int k_row = k_row_start; k_row < k_row_end; ++k_row) {
                for(int k_col = k_col_start; k_col < k_col_end; ++k_col) {
                    const int y = (k_row - row) + (kernel_size / 2);
                    const int x = (k_col - col) + (kernel_size / 2);
                    sum += img(k_row, k_col) * weights(y,x);
                }
            }
            dst(row,col) = sum;
        }
    }
    return 0;
}

template<class VT>
int draw_lines_in_img(Matrix<VT>& img, Matrix<VT>& line_img, Matrix<VT>& dst) {
    for(int row = 0; row < img.height; ++row){
        for(int col = 0; col < img.width; ++col){
            if(line_img(row,col) == 0){
                dst(row,col) = 1;
            } else {
                dst(row,col) = img(row,col);
            }
        }
    }

    Matrix<VT> blurred(img.height, img.width);
    
    gaus_blur(dst, 7, 1, blurred);
    
    for(int row = 0; row < img.height; ++row){
        for(int col = 0; col < img.width; ++col){
            dst(row,col) = blurred(row,col);
        }
    }
    
    return 0;
}

template<class VT>
int draw_lines_in_img(VT* img_buffer, const size_t height, const size_t width, VT* line_img_buffer, VT* dst_buffer) {
    Matrix<VT> img(img_buffer, height, width);
    Matrix<VT> line_img(line_img_buffer, height, width);
    Matrix<VT> dst(dst_buffer, height, width);
    return draw_lines_in_img(img, line_img, dst);
}


template<class ELEMENT_T>
std::vector<ELEMENT_T> flatten_2d_list(std::vector<std::vector<ELEMENT_T>> list){
    std::vector<ELEMENT_T> res;
    for(auto row : list){
        for(auto col : row){
            res.push_back(col);
        }   
    }
    return res;
}

std::tuple<int,int> get_overlap(int tile_size, int overlap, int size){
    int n_tiles = std::ceil(static_cast<float>(size - tile_size) / static_cast<float>(tile_size - overlap)) + 1;
    int overlap_new = (overlap + (((n_tiles * tile_size -  ((n_tiles - 1) * overlap)) - size) / (n_tiles-1)));
    return std::make_tuple(overlap_new, n_tiles);
}

template<class VT>
std::vector<std::vector<std::tuple<int,int>>> get_tiles(Matrix<VT>& img, int tile_size, int overlap){
    const int img_height = img.height;
    const int img_width = img.width;
    
    auto [overlap_y, tiles_y] = get_overlap(tile_size, overlap, img_height);
    auto [overlap_x, tiles_x] = get_overlap(tile_size, overlap, img_width);
    std::vector<std::vector<std::tuple<int,int>>> tiles;
    for(int row_ = 0; row_ < (int)(tiles_y * (tile_size - overlap_y)); row_ += tile_size - overlap_y) {
        
        int row = row_;
        if((row + tile_size) > img_height){
            row -= (row + tile_size) - img_height;
        }
        std::vector<std::tuple<int,int>> tile_row;
        for(int col_ = 0; col_ < (int)(tiles_x * (tile_size - overlap_x)); col_ += tile_size - overlap_x){
            int col = col_;
            if ((col + tile_size) > img_width){
                col -= (col + tile_size) - img_width;
            }
            tile_row.push_back(std::make_tuple(row, col));
        }
        tiles.push_back(tile_row);
    }
    return tiles;
}


auto split_2d_list(std::vector<std::vector<std::tuple<int,int>>> tiles, bool row_even, bool col_even){
    std::vector<std::tuple<int,int>> res;
    
    for(int row_i = 0; row_i < tiles.size(); ++row_i){
        auto row = tiles[row_i];
        if((row_i%2) != row_even) {
            continue;
        }
        for (int col_i = 0; col_i < row.size(); ++col_i){
            if((col_i%2) != col_even){
                continue;
            }
            res.push_back(row[col_i]);
        }
    }
    return res;
}

template<class VT, class DrawClass>
auto rasterize_img(Matrix<VT>& img, Matrix<VT>& line_img, VT max_val, DrawClass& draw_obj, int tile_row, int tile_col) {
    Matrix<VT> res(img.height, img.width);
    draw_obj.draw_in_img(img, line_img, res);
    const int pad = 10;
    std::vector<std::tuple<int,int>> all_coords(256*256);

    Matrix<VT> res_window(pad*2,pad*2);
    for(int i = 0; i < (img.height*img.width); ++i){
        
        Matrix<VT> padded_res = res(pad,-pad,pad,-pad);
        const std::tuple<int,int> coord = argmin(padded_res);
        const int row = std::get<0>(coord) + pad;
        const int col = std::get<1>(coord) + pad;

        const VT min_val = res(row,col);
        
        if(min_val > max_val){
            break;
        }

        all_coords.push_back(std::make_tuple(row,col));
    
        
        int coord_tmp[] = {row,col};



        draw_obj.draw(row, col, line_img, tile_row, tile_col);


        line_img(row,col) = 0;

        auto img_window = img(row-pad, row+pad, col-pad, col+pad); 
        auto line_img_window = line_img(row-pad,row+pad, col-pad, col+pad); 
        draw_obj.draw_in_img(img_window, line_img_window, res_window);
        
        
        auto res_window_a = res_window(pad/2, -pad/2, pad/2, -pad/2);
        auto res_window_b = res(row-pad/2, row+pad/2, col-pad/2, col+pad/2);
        
        for(int j = 0; j < pad*pad; ++j){
            res_window_b(j) = std::max(res_window_a(j),res_window_b(j));
        }
        

        res(row,col) = 1;

        
    }
    return all_coords;
}

template<class VT>
int rasterize_img(VT* img_buffer,  const size_t height, const size_t width, VT* line_img_buffer, VT max_val) {
    Matrix<VT> img(img_buffer, height, width);
    Matrix<VT> line_img(line_img_buffer, height, width);
    return 0;
}

template<class VT>
class DrawLine{
    public:
        const Matrix<VT>& img;
        Matrix<VT> grads_y;
        Matrix<VT> grads_x;
        Matrix<VT> angle;
        Matrix<VT> magnitude;
        float max_val;
        float marker_size;
    public:
        DrawLine(Matrix<VT>& img, float max_val, float marker_size=4): img{img}, grads_y{img.height, img.width}, grads_x{img.height, img.width}, angle{img.height, img.width}, magnitude{img.height, img.width}, max_val{max_val}, marker_size{marker_size}{
            Matrix<VT> img_smoothed(img.height, img.width, img.channels);
            gaus_blur(img, 5, 1, img_smoothed);
            sobel_filter(img_smoothed, 7, SOBEL_DIR_Y, grads_y);
            sobel_filter(img_smoothed, 7, SOBEL_DIR_X, grads_x);
            float mag_sum = 0;
            for(int row = 0; row < img.height; ++row){
                for(int col = 0; col < img.width; ++col) {
                    angle(row,col) = std::atan2(grads_x(row,col), -grads_y(row,col));
                    magnitude(row,col) = std::sqrt((grads_x(row, col)*grads_x(row, col)) + (grads_y(row, col)*grads_y(row, col)));
                    mag_sum += magnitude(row,col);
                }
            }

            mag_sum /= img.height * img.width * 0.7;

            for(int row = 0; row < img.height; ++row){
                for(int col = 0; col < img.width; ++col) {
                    magnitude(row,col) /= mag_sum;
                    if(magnitude(row,col) > 1){
                        magnitude(row,col) = 1;
                    }
                }
            }
        }

        int draw_in_img(Matrix<VT>& img, Matrix<VT>& line_img, Matrix<VT>& dst) {
            for(int row = 0; row < img.height; ++row){
                for(int col = 0; col < img.width; ++col){
                    if(line_img(row,col) == 0){
                        dst(row,col) = 2.5;
                    } else {
                        dst(row,col) = img(row,col);
                    }
                }
            }

            Matrix<VT> blurred(img.height, img.width);
            gaus_blur(dst, 7, 1, blurred);
            for(int row = 0; row < img.height; ++row){
                for(int col = 0; col < img.width; ++col){
                    dst(row,col) = blurred(row,col);
                }
            }
            
            return 0;
        }

        template<class VT_LINE>
        void draw(const int y, const int x, Matrix<VT_LINE>& line_img, int tile_row=0, int tile_col=0){

            float mag = img(y+tile_row,x+tile_col) / max_val;
            if(mag > 1){
                mag = 1;
            }
            float grad_mag = magnitude(y+tile_row,x+tile_col);

            mag = std::pow(1 - mag, 2);
            mag = ( (mag) * marker_size) + 0.5;

            float ang = (1-grad_mag)*PI*0.25 + grad_mag * angle(y+tile_row,x+tile_col);

            const double fx = line_img.width_buffer / img.width_buffer;
            const double fy = line_img.height_buffer / img.height_buffer;
            const double f = std::min(fx, fy);
            
            const int coord_tmp[] = {static_cast<int>(y*f), static_cast<int>(x*f)};

            draw_line_polar<VT_LINE>(line_img, coord_tmp, ang, mag*(f), 0); // 1+1.5
        }

};

template<class VT, class DrawClass>
std::vector<std::tuple<int,int>> rasterize_img_tiled(Matrix<VT>& img, Matrix<VT>& line_img, VT max_val, int tile_size, int overlap, DrawClass& draw_obj) {
    auto tiles_2d = get_tiles(img, tile_size, overlap);
    auto tiles_a = split_2d_list(tiles_2d, false, false);
    auto tiles_b = split_2d_list(tiles_2d, true, false);
    auto tiles_c = split_2d_list(tiles_2d, false, true);
    auto tiles_d = split_2d_list(tiles_2d, true, true);
    auto tile_sets = {tiles_a, tiles_b, tiles_c, tiles_d};  //
    const int padding = 20;
    
    Matrix<VT> img_tile(tile_size, tile_size); 
    Matrix<VT> line_img_tile(tile_size, tile_size); 
    std::vector<std::tuple<int,int>> all_coords;
    #pragma omp parallel
    {
        std::vector<std::tuple<int,int>> coords_per_thread;
        for(auto tiles : tile_sets){
            #pragma omp  for 
            for(int i = 0; i < tiles.size(); ++i){
                auto tile = tiles[i];
                int row = std::get<0>(tile);
                int col = std::get<1>(tile);
                
                
                auto img_tile = img(row, row+tile_size, col, col+tile_size);
                auto line_img_tile = line_img(row, row+tile_size, col, col+tile_size);                

                auto coords = rasterize_img(img_tile, line_img_tile, max_val, draw_obj, row, col);
                
                for(int y = 0; y < line_img_tile.height; ++y){
                    for(int x = 0; x < line_img_tile.width; ++x){
                        line_img_tile(y,x) = 1;
                        
                    }
                }
                
                for(auto coord : coords) {
                    int y = std::get<0>(coord);
                    int x = std::get<1>(coord);
                    if((y >= padding) && (y <= (tile_size - padding)) && (x >= padding) && (x <= (tile_size - padding))){
                        y += row;
                        x += col;
                        coords_per_thread.push_back(std::make_tuple(y,x));
                    }
                }
                
            }
            #pragma omp critical
            {
                for(auto coord : coords_per_thread) {
                    all_coords.push_back(coord);
                }
            }
            for(int y = 0; y < line_img.height; ++y){
                for(int x = 0; x < line_img.width; ++x){
                    line_img(y,x) = 1;
                }
            }
            for(auto coord : all_coords) {
                int coord_tmp[] = {std::get<0>(coord), std::get<1>(coord)};
                int y = std::get<0>(coord);
                int x = std::get<1>(coord);
                line_img(y,x) = 0;
                draw_obj.draw(y, x, line_img);
            }
                
        }
    }
    for(auto coord : all_coords) {
        int coord_tmp[] = {std::get<0>(coord), std::get<1>(coord)};
        int y = std::get<0>(coord);
        int x = std::get<1>(coord);
    }
    return all_coords;
};




template<class VT>
int rasterize_img_tiled(VT* img_buffer,  const size_t height, const size_t width, VT* line_img_buffer, VT max_val, int tile_size, int overlap) {
    Matrix<VT> img(img_buffer, height, width);
    Matrix<VT> line_img(line_img_buffer, height, width);
    DrawLine draw_obj(img, max_val);
    auto coords = rasterize_img_tiled(img, line_img, max_val, tile_size, overlap, draw_obj);
    return 0;
}


extern "C" {
    int argmin_float32(float*  buffer, const size_t height, const size_t width, int res[2]){
        return argmin<float>(buffer, height, width, res);
    }
    int argmin_uint8(uint8_t*  buffer, const size_t height, const size_t width, int res[2]){
        return argmin<uint8_t>(buffer, height, width, res);
    }
        
    int draw_line_polar_uint8(uint8_t* buffer, const size_t height, const size_t width, const int p[2], const float angle, const float mag, uint8_t color) {
        return draw_line_polar<uint8_t>(buffer, height,  width, p, angle, mag, color);
    }

    int draw_line_polar_float32(float* buffer, const size_t height, const size_t width, const int p[2], const float angle, const float mag, float color) {
        return draw_line_polar<float>(buffer, height,  width, p, angle, mag, color);
    }

    int draw_line_uint8(uint8_t*  buffer, const size_t height, const size_t width, const int p1[2], const int p2[2], uint8_t color){
        return draw_line<uint8_t>(buffer, height, width, p1, p2, color);
    }

    int draw_line_float32(float*  buffer, const size_t height, const size_t width, const int p1[2], const int p2[2], float color){
        return draw_line<float>(buffer, height, width, p1, p2, color);
    }

    int blur_float32(float* img, const size_t height, const size_t width, const int kernel_size, float* dst) {
        return blur<float>(img, height, width, kernel_size, dst);
    } 
    int blur_uint8(uint8_t* img, const size_t height, const size_t width, const int kernel_size, uint8_t* dst) {
        return blur<uint8_t>(img, height, width, kernel_size, dst);
    }

    int gaus_blur_float32(float* img, const size_t height, const size_t width, const int kernel_size, float std_dev, float* dst) {
        return gaus_blur<float>(img, height, width, kernel_size, std_dev, dst);
    } 
    int gaus_blur_uint8(uint8_t* img, const size_t height, const size_t width, const int kernel_size, float std_dev, uint8_t* dst) {
        return gaus_blur<uint8_t>(img, height, width, kernel_size, std_dev, dst);
    }

    int rasterize_img_float32(float* img_buffer,  const size_t height, const size_t width, float* line_img_buffer, float max_val){
        return rasterize_img(img_buffer,  height, width, line_img_buffer, max_val);
    }

    int draw_lines_in_img_float32(float* img_buffer, const size_t height, const size_t width, float* line_img_buffer, float* dst_buffer) {
        return draw_lines_in_img(img_buffer, height, width, line_img_buffer, dst_buffer);
    }

    int rasterize_img_tiled_float32(float* img_buffer,  const size_t height, const size_t width, float* line_img_buffer, float max_val, int tile_size, int overlap){
        return rasterize_img_tiled(img_buffer, height, width, line_img_buffer, max_val, tile_size, overlap);
    }

    void rasterize_img_tiled_uint8(uint8_t* img_buffer,  const size_t height, const size_t width, uint8_t* line_img_buffer,   const size_t height_line_img, const size_t width_line_img, float max_val, float marker_size, int tile_size, int overlap){
        Matrix<uint8_t> img(img_buffer, height, width, 4);
        Matrix<uint8_t> line_img(line_img_buffer, height_line_img, width_line_img, 4);

        Matrix<float> img_float(height, width);
        Matrix<float> line_img_float(height, width);

        float mean = 0;

        int bins[256] = {0};
        for(int i = 0; i < 256; ++i)
            bins[i] = 0;

        for(int row = 0; row < height; ++row){
            for(int col = 0; col < width; ++col) {
                img_float(row,col) = (0.3 * static_cast<float>(img(row, col, 0)) + 0.59 * static_cast<float>(img(row, col, 1)) + 0.11 * static_cast<float>(img(row, col, 2))) / 255;
                line_img_float(row,col) = 1;
                mean += img_float(row,col);
                bins[(int)(img_float(row,col)*255)]++;

                float diff = img_float(row,col) - 0.2;
                if(diff < 0){
                    diff = 0;
                }
            }
        }

        mean /= img.height*img.width;
        int cnt = 0;
        float median = 0;
        for(int i = 0; i < 256; ++i) {
            cnt += bins[i];
            if(cnt >= ((img.height*img.width)/2)) {
                median = i/255.0;
                break;
            }
        }

        DrawLine draw_obj(img_float, max_val, marker_size);
        auto coords = rasterize_img_tiled(img_float, line_img_float, max_val, tile_size, overlap, draw_obj);

        for(int row = 0; row < line_img.height; ++row){
            for(int col = 0; col < line_img.width; ++col) {
                line_img(row,col) = 255;
            }
        }
        for(auto coord : coords){
            const int y = std::get<0>(coord);
            const int x = std::get<1>(coord);
            draw_obj.draw(y, x, line_img);
        }
        
        Matrix<uint8_t> tmp(height_line_img, width_line_img);

        const double fx = width_line_img / width;
        const double fy = height_line_img / height;
        const double f = std::min(fx, fy);
            
        const int k = static_cast<int>((f-1) / 2) * 2 + 1;

        erode(line_img, 3, tmp);

        for(int row = 0; row < line_img.height; ++row){
            for(int col = 0; col < line_img.width; ++col) {
                uint8_t val = tmp(row,col);
                line_img(row,col, 0) = val;
                line_img(row,col, 1) = val;
                line_img(row,col, 2) = val;
                line_img(row,col, 3) = 255;
            }
        }

    }
    void rgb2gray(uint8_t* img_buffer,  const size_t height, const size_t width, uint8_t* line_img_buffer, float max_val, int tile_size, int overlap){
        Matrix<uint8_t> img(img_buffer, height, width, 4);
        Matrix<uint8_t> line_img(line_img_buffer, height, width, 4);

        Matrix<float> img_float(height, width);
        Matrix<float> line_img_float(height, width);

        for(int row = 0; row < height; ++row){
            for(int col = 0; col < width; ++col) {
                img_float(row,col) = (0.3 * static_cast<float>(img(row, col, 0)) + 0.59 * static_cast<float>(img(row, col, 1)) + 0.11 * static_cast<float>(img(row, col, 2))) / 255;
                float diff = img_float(row,col) - 0.1;
                if(diff < 0){
                    diff = 0;
                }
                img_float(row,col) /= (1+diff);
            }
        }

        gaus_blur(img_float, 3, 1, line_img_float);

        
        for(int row = 0; row < height; ++row){
            for(int col = 0; col < width; ++col) {
                uint8_t val = line_img_float(row,col)*255;
                line_img(row,col, 0) = val;
                line_img(row,col, 1) = val;
                line_img(row,col, 2) = val;
                line_img(row,col, 3) = 255;
            }
        }
        
    }

    int draw_circle_float32(float* img_buffer, int height, int width, int center[2], int r, float color) {
        return draw_circle_float32(img_buffer, height, width, center, r, color);
    }

}
