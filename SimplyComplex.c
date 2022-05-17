struct complex_int;
struct complex_float;
struct complex_double;
struct complex_int{
  int re, im;
  __host__ __device__ complex_int(void){}
  __host__ __device__ complex_int(const int x)
  {
    re = x;
    im = 0;
  }
  __host__ __device__ complex_int(const float x)
  {
    re = x;
    im = 0;
  }
  __host__ __device__ complex_int(const double x)
  {
    re = x;
    im = 0;
  }
  __host__ __device__ complex_int(const int a, const int b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_int(const int a, const float b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_int(const int a, const double b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_int(const float a, const int b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_int(const float a, const float b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_int(const float a, const double b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_int(const double a, const int b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_int(const double a, const float b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_int(const double a, const double b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_int(const complex_float &x);
  __host__ __device__ complex_int(const complex_double &x);
  __host__ __device__ complex_int& operator=(const int x)
  {
    re = x;
    im = 0;
    return *this;
  }
  __host__ __device__ complex_int& operator=(const float x)
  {
    re = x;
    im = 0;
    return *this;
  }
  __host__ __device__ complex_int& operator=(const double x)
  {
    re = x;
    im = 0;
    return *this;
  }
  __host__ __device__ complex_int& operator=(const complex_int &x)
  {
    re = x.re;
    im = x.im;
    return *this;
  }
  __host__ __device__ complex_int& operator=(const complex_float &x);
  __host__ __device__ complex_int& operator=(const complex_double &x);
  __host__ __device__ complex_int operator-(void) const
  {
    return complex_int(-re, -im);
  }
  __host__ __device__ complex_int& operator+=(const int x)
  {
    re += x;
    return *this;
  }
  __host__ __device__ complex_int& operator+=(const float x)
  {
    re += x;
    return *this;
  }
  __host__ __device__ complex_int& operator+=(const double x)
  {
    re += x;
    return *this;
  }
  __host__ __device__ complex_int& operator+=(const complex_int &x)
  {
    re += x.re;
    im += x.im;
    return *this;
  }
  __host__ __device__ complex_int& operator+=(const complex_float &x);
  __host__ __device__ complex_int& operator+=(const complex_double &x);
  __host__ __device__ complex_int& operator-=(const int x)
  {
    re -= x;
    return *this;
  }
  __host__ __device__ complex_int& operator-=(const float x)
  {
    re -= x;
    return *this;
  }
  __host__ __device__ complex_int& operator-=(const double x)
  {
    re -= x;
    return *this;
  }
  __host__ __device__ complex_int& operator-=(const complex_int &x)
  {
    re -= x.re;
    im -= x.im;
    return *this;
  }
  __host__ __device__ complex_int& operator-=(const complex_float &x);
  __host__ __device__ complex_int& operator-=(const complex_double &x);
  __host__ __device__ complex_int& operator*=(const int x)
  {
    re *= x;
    im *= x;
    return *this;
  }
  __host__ __device__ complex_int& operator*=(const float x)
  {
    re *= x;
    im *= x;
    return *this;
  }
  __host__ __device__ complex_int& operator*=(const double x)
  {
    re *= x;
    im *= x;
    return *this;
  }
  __host__ __device__ complex_int& operator*=(const complex_int &x)
  {
    int re_tmp, im_tmp;
    re_tmp = re*(x.re) - im*(x.im);
    im_tmp = re*(x.im) + im*(x.re);
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_int& operator*=(const complex_float &x);
  __host__ __device__ complex_int& operator*=(const complex_double &x);
  __host__ __device__ complex_int& operator/=(const int x)
  {
    re /= x;
    im /= x;
    return *this;
  }
  __host__ __device__ complex_int& operator/=(const float x)
  {
    re /= x;
    im /= x;
    return *this;
  }
  __host__ __device__ complex_int& operator/=(const double x)
  {
    re /= x;
    im /= x;
    return *this;
  }
  __host__ __device__ complex_int& operator/=(const complex_int &x)
  {
    int re_tmp, im_tmp;
    re_tmp = (re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    im_tmp = (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_int& operator/=(const complex_float &x);
  __host__ __device__ complex_int& operator/=(const complex_double &x);
  __host__ __device__ complex_int operator+(const int x) const
  {
    return complex_int(re+x, im);
  }
  __host__ __device__ complex_int operator+(const float x) const
  {
    return complex_int(re+x, im);
  }
  __host__ __device__ complex_int operator+(const double x) const
  {
    return complex_int(re+x, im);
  }
  __host__ __device__ complex_int operator+(const complex_int &x) const
  {
    return complex_int(re+x.re, im+x.im);
  }
  __host__ __device__ complex_int operator-(const int x) const
  {
    return complex_int(re-x, im);
  }
  __host__ __device__ complex_int operator-(const float x) const
  {
    return complex_int(re-x, im);
  }
  __host__ __device__ complex_int operator-(const double x) const
  {
    return complex_int(re-x, im);
  }
  __host__ __device__ complex_int operator-(const complex_int &x) const
  {
    return complex_int(re-x.re, im-x.im);
  }
  __host__ __device__ complex_int operator*(const int x) const
  {
    return complex_int(re*x, im*x);
  }
  __host__ __device__ complex_int operator*(const float x) const
  {
    return complex_int(re*x, im*x);
  }
  __host__ __device__ complex_int operator*(const double x) const
  {
    return complex_int(re*x, im*x);
  }
  __host__ __device__ complex_int operator*(const complex_int &x) const
  {
    return complex_int(re*(x.re) - im*(x.im), re*(x.im) + im*(x.re));
  }
  __host__ __device__ complex_int operator/(const int x) const
  {
    return complex_int(re/x, im/x);
  }
  __host__ __device__ complex_int operator/(const float x) const
  {
    return complex_int(re/x, im/x);
  }
  __host__ __device__ complex_int operator/(const double x) const
  {
    return complex_int(re/x, im/x);
  }
  __host__ __device__ complex_int operator/(const complex_int &x) const
  {
    return complex_int((re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)), (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)));
  }
};
struct complex_float{
  float re, im;
  __host__ __device__ complex_float(void){}
  __host__ __device__ complex_float(const int x)
  {
    re = x;
    im = 0;
  }
  __host__ __device__ complex_float(const float x)
  {
    re = x;
    im = 0;
  }
  __host__ __device__ complex_float(const double x)
  {
    re = x;
    im = 0;
  }
  __host__ __device__ complex_float(const int a, const int b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_float(const int a, const float b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_float(const int a, const double b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_float(const float a, const int b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_float(const float a, const float b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_float(const float a, const double b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_float(const double a, const int b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_float(const double a, const float b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_float(const double a, const double b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_float(const complex_int &x)
  {
    re = x.re;
    im = x.im;
  }
  __host__ __device__ complex_float(const complex_double &x);
  __host__ __device__ complex_float& operator=(const int x)
  {
    re = x;
    im = 0;
    return *this;
  }
  __host__ __device__ complex_float& operator=(const float x)
  {
    re = x;
    im = 0;
    return *this;
  }
  __host__ __device__ complex_float& operator=(const double x)
  {
    re = x;
    im = 0;
    return *this;
  }
  __host__ __device__ complex_float& operator=(const complex_int &x)
  {
    re = x.re;
    im = x.im;
    return *this;
  }
  __host__ __device__ complex_float& operator=(const complex_float &x)
  {
    re = x.re;
    im = x.im;
    return *this;
  }
  __host__ __device__ complex_float& operator=(const complex_double &x);
  __host__ __device__ complex_float operator-(void) const
  {
    return complex_float(-re, -im);
  }
  __host__ __device__ complex_float& operator+=(const int x)
  {
    re += x;
    return *this;
  }
  __host__ __device__ complex_float& operator+=(const float x)
  {
    re += x;
    return *this;
  }
  __host__ __device__ complex_float& operator+=(const double x)
  {
    re += x;
    return *this;
  }
  __host__ __device__ complex_float& operator+=(const complex_int &x)
  {
    re += x.re;
    im += x.im;
    return *this;
  }
  __host__ __device__ complex_float& operator+=(const complex_float &x)
  {
    re += x.re;
    im += x.im;
    return *this;
  }
  __host__ __device__ complex_float& operator+=(const complex_double &x);
  __host__ __device__ complex_float& operator-=(const int x)
  {
    re -= x;
    return *this;
  }
  __host__ __device__ complex_float& operator-=(const float x)
  {
    re -= x;
    return *this;
  }
  __host__ __device__ complex_float& operator-=(const double x)
  {
    re -= x;
    return *this;
  }
  __host__ __device__ complex_float& operator-=(const complex_int &x)
  {
    re -= x.re;
    im -= x.im;
    return *this;
  }
  __host__ __device__ complex_float& operator-=(const complex_float &x)
  {
    re -= x.re;
    im -= x.im;
    return *this;
  }
  __host__ __device__ complex_float& operator-=(const complex_double &x);
  __host__ __device__ complex_float& operator*=(const int x)
  {
    re *= x;
    im *= x;
    return *this;
  }
  __host__ __device__ complex_float& operator*=(const float x)
  {
    re *= x;
    im *= x;
    return *this;
  }
  __host__ __device__ complex_float& operator*=(const double x)
  {
    re *= x;
    im *= x;
    return *this;
  }
  __host__ __device__ complex_float& operator*=(const complex_int &x)
  {
    float re_tmp, im_tmp;
    re_tmp = re*(x.re) - im*(x.im);
    im_tmp = re*(x.im) + im*(x.re);
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_float& operator*=(const complex_float &x)
  {
    float re_tmp, im_tmp;
    re_tmp = re*(x.re) - im*(x.im);
    im_tmp = re*(x.im) + im*(x.re);
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_float& operator*=(const complex_double &x);
  __host__ __device__ complex_float& operator/=(const int x)
  {
    re /= x;
    im /= x;
    return *this;
  }
  __host__ __device__ complex_float& operator/=(const float x)
  {
    re /= x;
    im /= x;
    return *this;
  }
  __host__ __device__ complex_float& operator/=(const double x)
  {
    re /= x;
    im /= x;
    return *this;
  }
  __host__ __device__ complex_float& operator/=(const complex_int &x)
  {
    float re_tmp, im_tmp;
    re_tmp = (re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    im_tmp = (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_float& operator/=(const complex_float &x)
  {
    float re_tmp, im_tmp;
    re_tmp = (re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    im_tmp = (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_float& operator/=(const complex_double &x);
  __host__ __device__ complex_float operator+(const int x) const
  {
    return complex_float(re+x, im);
  }
  __host__ __device__ complex_float operator+(const float x) const
  {
    return complex_float(re+x, im);
  }
  __host__ __device__ complex_float operator+(const double x) const
  {
    return complex_float(re+x, im);
  }
  __host__ __device__ complex_float operator+(const complex_int &x) const
  {
    return complex_float(re+x.re, im+x.im);
  }
  __host__ __device__ complex_float operator+(const complex_float &x) const
  {
    return complex_float(re+x.re, im+x.im);
  }
  __host__ __device__ complex_float operator-(const int x) const
  {
    return complex_float(re-x, im);
  }
  __host__ __device__ complex_float operator-(const float x) const
  {
    return complex_float(re-x, im);
  }
  __host__ __device__ complex_float operator-(const double x) const
  {
    return complex_float(re-x, im);
  }
  __host__ __device__ complex_float operator-(const complex_int &x) const
  {
    return complex_float(re-x.re, im-x.im);
  }
  __host__ __device__ complex_float operator-(const complex_float &x) const
  {
    return complex_float(re-x.re, im-x.im);
  }
  __host__ __device__ complex_float operator*(const int x) const
  {
    return complex_float(re*x, im*x);
  }
  __host__ __device__ complex_float operator*(const float x) const
  {
    return complex_float(re*x, im*x);
  }
  __host__ __device__ complex_float operator*(const double x) const
  {
    return complex_float(re*x, im*x);
  }
  __host__ __device__ complex_float operator*(const complex_int &x) const
  {
    return complex_float(re*(x.re) - im*(x.im), re*(x.im) + im*(x.re));
  }
  __host__ __device__ complex_float operator*(const complex_float &x) const
  {
    return complex_float(re*(x.re) - im*(x.im), re*(x.im) + im*(x.re));
  }
  __host__ __device__ complex_float operator/(const int x) const
  {
    return complex_float(re/x, im/x);
  }
  __host__ __device__ complex_float operator/(const float x) const
  {
    return complex_float(re/x, im/x);
  }
  __host__ __device__ complex_float operator/(const double x) const
  {
    return complex_float(re/x, im/x);
  }
  __host__ __device__ complex_float operator/(const complex_int &x) const
  {
    return complex_float((re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)), (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)));
  }
  __host__ __device__ complex_float operator/(const complex_float &x) const
  {
    return complex_float((re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)), (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)));
  }
};
struct complex_double{
  double re, im;
  __host__ __device__ complex_double(void){}
  __host__ __device__ complex_double(const int x)
  {
    re = x;
    im = 0;
  }
  __host__ __device__ complex_double(const float x)
  {
    re = x;
    im = 0;
  }
  __host__ __device__ complex_double(const double x)
  {
    re = x;
    im = 0;
  }
  __host__ __device__ complex_double(const int a, const int b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_double(const int a, const float b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_double(const int a, const double b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_double(const float a, const int b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_double(const float a, const float b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_double(const float a, const double b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_double(const double a, const int b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_double(const double a, const float b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_double(const double a, const double b)
  {
    re = a;
    im = b;
  }
  __host__ __device__ complex_double(const complex_int &x)
  {
    re = x.re;
    im = x.im;
  }
  __host__ __device__ complex_double(const complex_float &x)
  {
    re = x.re;
    im = x.im;
  }
  __host__ __device__ complex_double& operator=(const int x)
  {
    re = x;
    im = 0;
    return *this;
  }
  __host__ __device__ complex_double& operator=(const float x)
  {
    re = x;
    im = 0;
    return *this;
  }
  __host__ __device__ complex_double& operator=(const double x)
  {
    re = x;
    im = 0;
    return *this;
  }
  __host__ __device__ complex_double& operator=(const complex_int &x)
  {
    re = x.re;
    im = x.im;
    return *this;
  }
  __host__ __device__ complex_double& operator=(const complex_float &x)
  {
    re = x.re;
    im = x.im;
    return *this;
  }
  __host__ __device__ complex_double& operator=(const complex_double &x)
  {
    re = x.re;
    im = x.im;
    return *this;
  }
  __host__ __device__ complex_double operator-(void) const
  {
    return complex_double(-re, -im);
  }
  __host__ __device__ complex_double& operator+=(const int x)
  {
    re += x;
    return *this;
  }
  __host__ __device__ complex_double& operator+=(const float x)
  {
    re += x;
    return *this;
  }
  __host__ __device__ complex_double& operator+=(const double x)
  {
    re += x;
    return *this;
  }
  __host__ __device__ complex_double& operator+=(const complex_int &x)
  {
    re += x.re;
    im += x.im;
    return *this;
  }
  __host__ __device__ complex_double& operator+=(const complex_float &x)
  {
    re += x.re;
    im += x.im;
    return *this;
  }
  __host__ __device__ complex_double& operator+=(const complex_double &x)
  {
    re += x.re;
    im += x.im;
    return *this;
  }
  __host__ __device__ complex_double& operator-=(const int x)
  {
    re -= x;
    return *this;
  }
  __host__ __device__ complex_double& operator-=(const float x)
  {
    re -= x;
    return *this;
  }
  __host__ __device__ complex_double& operator-=(const double x)
  {
    re -= x;
    return *this;
  }
  __host__ __device__ complex_double& operator-=(const complex_int &x)
  {
    re -= x.re;
    im -= x.im;
    return *this;
  }
  __host__ __device__ complex_double& operator-=(const complex_float &x)
  {
    re -= x.re;
    im -= x.im;
    return *this;
  }
  __host__ __device__ complex_double& operator-=(const complex_double &x)
  {
    re -= x.re;
    im -= x.im;
    return *this;
  }
  __host__ __device__ complex_double& operator*=(const int x)
  {
    re *= x;
    im *= x;
    return *this;
  }
  __host__ __device__ complex_double& operator*=(const float x)
  {
    re *= x;
    im *= x;
    return *this;
  }
  __host__ __device__ complex_double& operator*=(const double x)
  {
    re *= x;
    im *= x;
    return *this;
  }
  __host__ __device__ complex_double& operator*=(const complex_int &x)
  {
    double re_tmp, im_tmp;
    re_tmp = re*(x.re) - im*(x.im);
    im_tmp = re*(x.im) + im*(x.re);
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_double& operator*=(const complex_float &x)
  {
    double re_tmp, im_tmp;
    re_tmp = re*(x.re) - im*(x.im);
    im_tmp = re*(x.im) + im*(x.re);
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_double& operator*=(const complex_double &x)
  {
    double re_tmp, im_tmp;
    re_tmp = re*(x.re) - im*(x.im);
    im_tmp = re*(x.im) + im*(x.re);
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_double& operator/=(const int x)
  {
    re /= x;
    im /= x;
    return *this;
  }
  __host__ __device__ complex_double& operator/=(const float x)
  {
    re /= x;
    im /= x;
    return *this;
  }
  __host__ __device__ complex_double& operator/=(const double x)
  {
    re /= x;
    im /= x;
    return *this;
  }
  __host__ __device__ complex_double& operator/=(const complex_int &x)
  {
    double re_tmp, im_tmp;
    re_tmp = (re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    im_tmp = (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_double& operator/=(const complex_float &x)
  {
    double re_tmp, im_tmp;
    re_tmp = (re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    im_tmp = (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_double& operator/=(const complex_double &x)
  {
    double re_tmp, im_tmp;
    re_tmp = (re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    im_tmp = (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
    re = re_tmp;
    im = im_tmp;
    return *this;
  }
  __host__ __device__ complex_double operator+(const int x) const
  {
    return complex_double(re+x, im);
  }
  __host__ __device__ complex_double operator+(const float x) const
  {
    return complex_double(re+x, im);
  }
  __host__ __device__ complex_double operator+(const double x) const
  {
    return complex_double(re+x, im);
  }
  __host__ __device__ complex_double operator+(const complex_int &x) const
  {
    return complex_double(re+x.re, im+x.im);
  }
  __host__ __device__ complex_double operator+(const complex_float &x) const
  {
    return complex_double(re+x.re, im+x.im);
  }
  __host__ __device__ complex_double operator+(const complex_double &x) const
  {
    return complex_double(re+x.re, im+x.im);
  }
  __host__ __device__ complex_double operator-(const int x) const
  {
    return complex_double(re-x, im);
  }
  __host__ __device__ complex_double operator-(const float x) const
  {
    return complex_double(re-x, im);
  }
  __host__ __device__ complex_double operator-(const double x) const
  {
    return complex_double(re-x, im);
  }
  __host__ __device__ complex_double operator-(const complex_int &x) const
  {
    return complex_double(re-x.re, im-x.im);
  }
  __host__ __device__ complex_double operator-(const complex_float &x) const
  {
    return complex_double(re-x.re, im-x.im);
  }
  __host__ __device__ complex_double operator-(const complex_double &x) const
  {
    return complex_double(re-x.re, im-x.im);
  }
  __host__ __device__ complex_double operator*(const int x) const
  {
    return complex_double(re*x, im*x);
  }
  __host__ __device__ complex_double operator*(const float x) const
  {
    return complex_double(re*x, im*x);
  }
  __host__ __device__ complex_double operator*(const double x) const
  {
    return complex_double(re*x, im*x);
  }
  __host__ __device__ complex_double operator*(const complex_int &x) const
  {
    return complex_double(re*(x.re) - im*(x.im), re*(x.im) + im*(x.re));
  }
  __host__ __device__ complex_double operator*(const complex_float &x) const
  {
    return complex_double(re*(x.re) - im*(x.im), re*(x.im) + im*(x.re));
  }
  __host__ __device__ complex_double operator*(const complex_double &x) const
  {
    return complex_double(re*(x.re) - im*(x.im), re*(x.im) + im*(x.re));
  }
  __host__ __device__ complex_double operator/(const int x) const
  {
    return complex_double(re/x, im/x);
  }
  __host__ __device__ complex_double operator/(const float x) const
  {
    return complex_double(re/x, im/x);
  }
  __host__ __device__ complex_double operator/(const double x) const
  {
    return complex_double(re/x, im/x);
  }
  __host__ __device__ complex_double operator/(const complex_int &x) const
  {
    return complex_double((re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)), (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)));
  }
  __host__ __device__ complex_double operator/(const complex_float &x) const
  {
    return complex_double((re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)), (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)));
  }
  __host__ __device__ complex_double operator/(const complex_double &x) const
  {
    return complex_double((re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)), (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im)));
  }
};
__host__ __device__ complex_int operator+(const int x, const complex_int &c){return c+x;}
__host__ __device__ complex_int operator+(const float x, const complex_int &c){return c+x;}
__host__ __device__ complex_int operator+(const double x, const complex_int &c){return c+x;}
__host__ __device__ complex_int operator-(const int x, const complex_int &c){return -c+x;}
__host__ __device__ complex_int operator-(const float x, const complex_int &c){return -c+x;}
__host__ __device__ complex_int operator-(const double x, const complex_int &c){return -c+x;}
__host__ __device__ complex_int operator*(const int x, const complex_int &c){return c*x;}
__host__ __device__ complex_int operator*(const float x, const complex_int &c){return c*x;}
__host__ __device__ complex_int operator*(const double x, const complex_int &c){return c*x;}
__host__ __device__ complex_int operator/(const int x, const complex_int &c){return ((complex_int)x)/c;}
__host__ __device__ complex_int operator/(const float x, const complex_int &c){return ((complex_int)x)/c;}
__host__ __device__ complex_int operator/(const double x, const complex_int &c){return ((complex_int)x)/c;}
__host__ __device__ complex_float operator+(const int x, const complex_float &c){return c+x;}
__host__ __device__ complex_float operator+(const float x, const complex_float &c){return c+x;}
__host__ __device__ complex_float operator+(const double x, const complex_float &c){return c+x;}
__host__ __device__ complex_float operator+(const complex_int &x, const complex_float &c){return c+x;}
__host__ __device__ complex_float operator-(const int x, const complex_float &c){return -c+x;}
__host__ __device__ complex_float operator-(const float x, const complex_float &c){return -c+x;}
__host__ __device__ complex_float operator-(const double x, const complex_float &c){return -c+x;}
__host__ __device__ complex_float operator-(const complex_int &x, const complex_float &c){return -c+x;}
__host__ __device__ complex_float operator*(const int x, const complex_float &c){return c*x;}
__host__ __device__ complex_float operator*(const float x, const complex_float &c){return c*x;}
__host__ __device__ complex_float operator*(const double x, const complex_float &c){return c*x;}
__host__ __device__ complex_float operator*(const complex_int &x, const complex_float &c){return c*x;}
__host__ __device__ complex_float operator/(const int x, const complex_float &c){return ((complex_float)x)/c;}
__host__ __device__ complex_float operator/(const float x, const complex_float &c){return ((complex_float)x)/c;}
__host__ __device__ complex_float operator/(const double x, const complex_float &c){return ((complex_float)x)/c;}
__host__ __device__ complex_float operator/(const complex_int &x, const complex_float &c){return ((complex_float)x)/c;}
__host__ __device__ complex_double operator+(const int x, const complex_double &c){return c+x;}
__host__ __device__ complex_double operator+(const float x, const complex_double &c){return c+x;}
__host__ __device__ complex_double operator+(const double x, const complex_double &c){return c+x;}
__host__ __device__ complex_double operator+(const complex_int &x, const complex_double &c){return c+x;}
__host__ __device__ complex_double operator+(const complex_float &x, const complex_double &c){return c+x;}
__host__ __device__ complex_double operator-(const int x, const complex_double &c){return -c+x;}
__host__ __device__ complex_double operator-(const float x, const complex_double &c){return -c+x;}
__host__ __device__ complex_double operator-(const double x, const complex_double &c){return -c+x;}
__host__ __device__ complex_double operator-(const complex_int &x, const complex_double &c){return -c+x;}
__host__ __device__ complex_double operator-(const complex_float &x, const complex_double &c){return -c+x;}
__host__ __device__ complex_double operator*(const int x, const complex_double &c){return c*x;}
__host__ __device__ complex_double operator*(const float x, const complex_double &c){return c*x;}
__host__ __device__ complex_double operator*(const double x, const complex_double &c){return c*x;}
__host__ __device__ complex_double operator*(const complex_int &x, const complex_double &c){return c*x;}
__host__ __device__ complex_double operator*(const complex_float &x, const complex_double &c){return c*x;}
__host__ __device__ complex_double operator/(const int x, const complex_double &c){return ((complex_double)x)/c;}
__host__ __device__ complex_double operator/(const float x, const complex_double &c){return ((complex_double)x)/c;}
__host__ __device__ complex_double operator/(const double x, const complex_double &c){return ((complex_double)x)/c;}
__host__ __device__ complex_double operator/(const complex_int &x, const complex_double &c){return ((complex_double)x)/c;}
__host__ __device__ complex_double operator/(const complex_float &x, const complex_double &c){return ((complex_double)x)/c;}
__host__ __device__  complex_int::complex_int(const complex_float &x)
{
  re = x.re;
  im = x.im;
}
__host__ __device__  complex_int::complex_int(const complex_double &x)
{
  re = x.re;
  im = x.im;
}
__host__ __device__ complex_int& complex_int::operator=(const complex_float &x)
{
  re = x.re;
  im = x.im;
  return *this;
}
__host__ __device__ complex_int& complex_int::operator=(const complex_double &x)
{
  re = x.re;
  im = x.im;
  return *this;
}
__host__ __device__ complex_int& complex_int::operator+=(const complex_float &x)
{
  re += x.re;
  im += x.im;
  return *this;
}
__host__ __device__ complex_int& complex_int::operator+=(const complex_double &x)
{
  re += x.re;
  im += x.im;
  return *this;
}
__host__ __device__ complex_int& complex_int::operator-=(const complex_float &x)
{
  re -= x.re;
  im -= x.im;
  return *this;
}
__host__ __device__ complex_int& complex_int::operator-=(const complex_double &x)
{
  re -= x.re;
  im -= x.im;
  return *this;
}
__host__ __device__ complex_int& complex_int::operator*=(const complex_float &x)
{
  int re_tmp, im_tmp;
  re_tmp = re*(x.re) - im*(x.im);
  im_tmp = re*(x.im) + im*(x.re);
  re = re_tmp;
  im = im_tmp;
  return *this;
}
__host__ __device__ complex_int& complex_int::operator*=(const complex_double &x)
{
  int re_tmp, im_tmp;
  re_tmp = re*(x.re) - im*(x.im);
  im_tmp = re*(x.im) + im*(x.re);
  re = re_tmp;
  im = im_tmp;
  return *this;
}
__host__ __device__ complex_int& complex_int::operator/=(const complex_float &x)
{
  int re_tmp, im_tmp;
  re_tmp = (re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
  im_tmp = (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
  re = re_tmp;
  im = im_tmp;
  return *this;
}
__host__ __device__ complex_int& complex_int::operator/=(const complex_double &x)
{
  int re_tmp, im_tmp;
  re_tmp = (re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
  im_tmp = (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
  re = re_tmp;
  im = im_tmp;
  return *this;
}
__host__ __device__  complex_float::complex_float(const complex_double &x)
{
  re = x.re;
  im = x.im;
}
__host__ __device__ complex_float& complex_float::operator=(const complex_double &x)
{
  re = x.re;
  im = x.im;
  return *this;
}
__host__ __device__ complex_float& complex_float::operator+=(const complex_double &x)
{
  re += x.re;
  im += x.im;
  return *this;
}
__host__ __device__ complex_float& complex_float::operator-=(const complex_double &x)
{
  re -= x.re;
  im -= x.im;
  return *this;
}
__host__ __device__ complex_float& complex_float::operator*=(const complex_double &x)
{
  float re_tmp, im_tmp;
  re_tmp = re*(x.re) - im*(x.im);
  im_tmp = re*(x.im) + im*(x.re);
  re = re_tmp;
  im = im_tmp;
  return *this;
}
__host__ __device__ complex_float& complex_float::operator/=(const complex_double &x)
{
  float re_tmp, im_tmp;
  re_tmp = (re*(x.re) + im*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
  im_tmp = (im*(x.re) - re*(x.im)) / ((x.re)*(x.re) + (x.im)*(x.im));
  re = re_tmp;
  im = im_tmp;
  return *this;
}
