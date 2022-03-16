#include <gtest/gtest.h>
#include "SimplyComplex.h"
#include <iostream>

TEST(ComplexDouble, overload_plus_rhs_double){
    complex_double c = complex_double(1.1, 2.2);
    complex_double a;
    double b = 3.3;
    a = c + b;
    ASSERT_DOUBLE_EQ(4.4, a.re);
    ASSERT_DOUBLE_EQ(2.2, a.im);
    ASSERT_DOUBLE_EQ(1.1, c.re);
    ASSERT_DOUBLE_EQ(2.2, c.im);
    ASSERT_DOUBLE_EQ(3.3, b);
}
TEST(ComplexDouble, overload_plus_lhs_double){
    complex_double c = complex_double(1.1, 2.2);
    complex_double a;
    double b = 3.3;
    a = b + c;
    ASSERT_DOUBLE_EQ(4.4, a.re);
    ASSERT_DOUBLE_EQ(2.2, a.im);
    ASSERT_DOUBLE_EQ(1.1, c.re);
    ASSERT_DOUBLE_EQ(2.2, c.im);
    ASSERT_DOUBLE_EQ(3.3, b);
}
TEST(ComplexDouble, overload_prod_rhs_double){
    complex_double c = complex_double(1.1, 2.2);
    complex_double a;
    double b = 3.3;
    a = c*b;
    ASSERT_DOUBLE_EQ(3.63, a.re);
    ASSERT_DOUBLE_EQ(7.26, a.im);
    ASSERT_DOUBLE_EQ(1.1, c.re);
    ASSERT_DOUBLE_EQ(2.2, c.im);
    ASSERT_DOUBLE_EQ(3.3, b);
}
TEST(ComplexDouble, overload_prod_lhs_double){
    complex_double c = complex_double(1.1, 2.2);
    complex_double a;
    double b = 3.3;
    a = b*c;
    ASSERT_DOUBLE_EQ(3.63, a.re);
    ASSERT_DOUBLE_EQ(7.26, a.im);
    ASSERT_DOUBLE_EQ(1.1, c.re);
    ASSERT_DOUBLE_EQ(2.2, c.im);
    ASSERT_DOUBLE_EQ(3.3, b);
}
TEST(ComplexDouble, overload_div_rhs_double){
    complex_double c = complex_double(1.1, 2.2);
    complex_double a;
    double b = 3.3;
    a = c/b;
    ASSERT_DOUBLE_EQ(1.0/3.0, a.re);
    ASSERT_DOUBLE_EQ(2.0/3.0, a.im);
    ASSERT_DOUBLE_EQ(1.1, c.re);
    ASSERT_DOUBLE_EQ(2.2, c.im);
    ASSERT_DOUBLE_EQ(3.3, b);
}
TEST(ComplexDouble, overload_div_lhs_double){
    complex_double c = complex_double(1.1, 2.2);
    complex_double a;
    double b = 3.3;
    a = b/c;
    ASSERT_DOUBLE_EQ(0.6, a.re);
    ASSERT_DOUBLE_EQ(-1.2, a.im);
    ASSERT_DOUBLE_EQ(1.1, c.re);
    ASSERT_DOUBLE_EQ(2.2, c.im);
    ASSERT_DOUBLE_EQ(3.3, b);
}
TEST(ComplexDouble, overload_plus_complex_double){
    complex_double c = complex_double(1.1, 2.2);
    complex_double b = complex_double(3.3, 4.4);
    complex_double a;
    a = b + c;
    ASSERT_DOUBLE_EQ(4.4, a.re);
    ASSERT_DOUBLE_EQ(6.6, a.im);
    ASSERT_DOUBLE_EQ(1.1, c.re);
    ASSERT_DOUBLE_EQ(2.2, c.im);
    ASSERT_DOUBLE_EQ(3.3, b.re);
    ASSERT_DOUBLE_EQ(4.4, b.im);
}
TEST(ComplexDouble, overload_minus_complex_double){
    complex_double c = complex_double(1.1, 2.2);
    complex_double b = complex_double(3.3, 4.4);
    complex_double a;
    a = b - c;
    ASSERT_DOUBLE_EQ(2.2, a.re);
    ASSERT_DOUBLE_EQ(2.2, a.im);
    ASSERT_DOUBLE_EQ(1.1, c.re);
    ASSERT_DOUBLE_EQ(2.2, c.im);
    ASSERT_DOUBLE_EQ(3.3, b.re);
    ASSERT_DOUBLE_EQ(4.4, b.im);
}
TEST(ComplexDouble, overload_prod_complex_double){
    complex_double c = complex_double(1.1, 2.2);
    complex_double b = complex_double(3.3, 4.4);
    complex_double a;
    a = b*c;
    ASSERT_DOUBLE_EQ(-6.05, a.re);
    ASSERT_DOUBLE_EQ(12.1, a.im);
    ASSERT_DOUBLE_EQ(1.1, c.re);
    ASSERT_DOUBLE_EQ(2.2, c.im);
    ASSERT_DOUBLE_EQ(3.3, b.re);
    ASSERT_DOUBLE_EQ(4.4, b.im);
}
TEST(ComplexDouble, overload_div_complex_double){
    complex_double c = complex_double(1.1, 2.2);
    complex_double b = complex_double(3.3, 4.4);
    complex_double a;
    a = b/c;
    ASSERT_DOUBLE_EQ(2.2, a.re);
    ASSERT_DOUBLE_EQ(-0.4, a.im);
    ASSERT_DOUBLE_EQ(1.1, c.re);
    ASSERT_DOUBLE_EQ(2.2, c.im);
    ASSERT_DOUBLE_EQ(3.3, b.re);
    ASSERT_DOUBLE_EQ(4.4, b.im);
}
TEST(ComplexDouble, negate){
    complex_double b = complex_double(1.1, 2.2);
    complex_double a;
    a = -b;
    ASSERT_DOUBLE_EQ(-1.1, a.re);
    ASSERT_DOUBLE_EQ(-2.2, a.im);
    ASSERT_DOUBLE_EQ(1.1, b.re);
    ASSERT_DOUBLE_EQ(2.2, b.im);
}
TEST(ComplexDouble, cast_double){
    complex_double a;
    double b = 1.1;
    a = (complex_double)b;
    ASSERT_DOUBLE_EQ(1.1, a.re);
    ASSERT_DOUBLE_EQ(0.0, a.im);
    ASSERT_DOUBLE_EQ(1.1, b);
}
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
