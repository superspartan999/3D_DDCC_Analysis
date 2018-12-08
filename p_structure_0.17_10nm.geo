/////GeneratedbyPython
/////ClaytonQwah2018

la = 2*10^-6; // 10 nm
u = 10^-4; // um 
n = 10^-7; // nm 
a = 10^-8; // a 
length = 30 *n; // nm 
mesh_x = 0.6 * n; 
tf = length/mesh_x; 

tf_qw = 15; 
tf_cap = 10; 
// // // // // // // // // // // // // // // // // // // // // // 
                    // // SUBSTRATE LAYER // // // / 
// // // // //// // // // // // // // // // // // // // // // // 

// // Define Points // // // / 

p1 = newp; Point (p1) = {0, 0, 0, la};
p2 = newp; Point (p2) = {length, 0, 0, la};
p3 = newp; Point (p3) = {length, length, 0, la};
p4 = newp; Point (p4) = {0, length, 0, la};

// // Define Line // // // /

l1 = newl; Line (l1) = {p1, p2};
l2 = newl; Line (l2) = {p2, p3};
l3 = newl; Line (l3) = {p3, p4};
l4 = newl; Line (l4) = {p4, p1};

ll1 = newll; Line Loop (ll1) = {l1, l2, l3, l4};
ps1 = news; Plane Surface (ps1) = {ll1};

// // Define Transfinite Mesh // // // /

Transfinite Line {l1} = tf + 1;
Transfinite Line {l2} = tf + 1;
Transfinite Line {l3} = tf + 1;
Transfinite Line {l4} = tf + 1;

Transfinite Surface {ps1} = {1, 2, 3, 4};
// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 2 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p11 = newp;Point(p11) = {0, 0, p11*n, la};
p12 = newp;Point(p12 ) = {length, 0,p12*n, la};
p13 = newp;Point(p13) = {length, length,p13*n, la};
p14 = newp;Point(p14) = {0, length, p14*n, la};

// // Define Lines and Volume /// 

l11 = newl;Line(l11)=l11,l12};
l12 = newl;Line(l12)=l12,l13};
l13 = newl;Line(l13)=l13,l14};
l14 = newl;Line(l14)=l14,l11};
l15 = newl;Line(l15)=l1,l11};
l16 = newl;Line(l16)=l2,l12};
l17 = newl;Line(l17)=l3,l13};
l18 = newl;Line(l18)=l4,l14};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 2 // 
// // // // // // // // // // // // //

Transfinite Line {l11} = tf + 1;
Transfinite Line {l12} = tf + 1;
Transfinite Line {l13} = tf + 1;
Transfinite Line {l14} = tf;
Transfinite Line {l15} = 37.0 Using Progression 1.078;
Transfinite Line {l16} = 37.0 Using Progression 1.078;
Transfinite Line {l17} = 37.0 Using Progression 1.078;
Transfinite Line {l18} = 37.0 Using Progression 1.078;


// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 3 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p21 = newp;Point(p21) = {0, 0, p21*n, la};
p22 = newp;Point(p22 ) = {length, 0,p22*n, la};
p23 = newp;Point(p23) = {length, length,p23*n, la};
p24 = newp;Point(p24) = {0, length, p24*n, la};

// // Define Lines and Volume /// 

l21 = newl;Line(l21)=l21,l22};
l22 = newl;Line(l22)=l22,l23};
l23 = newl;Line(l23)=l23,l24};
l24 = newl;Line(l24)=l24,l21};
l25 = newl;Line(l25)=l11,l21};
l26 = newl;Line(l26)=l12,l22};
l27 = newl;Line(l27)=l13,l23};
l28 = newl;Line(l28)=l14,l24};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 3 // 
// // // // // // // // // // // // //

Transfinite Line {l21} = tf + 1;
Transfinite Line {l22} = tf + 1;
Transfinite Line {l23} = tf + 1;
Transfinite Line {l24} = tf;
Transfinite Line {l25} = 37.0 Using Progression 0.928;
Transfinite Line {l26} = 37.0 Using Progression 0.928;
Transfinite Line {l27} = 37.0 Using Progression 0.928;
Transfinite Line {l28} = 37.0 Using Progression 0.928;


// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 4 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p31 = newp;Point(p31) = {0, 0, p31*n, la};
p32 = newp;Point(p32 ) = {length, 0,p32*n, la};
p33 = newp;Point(p33) = {length, length,p33*n, la};
p34 = newp;Point(p34) = {0, length, p34*n, la};

// // Define Lines and Volume /// 

l31 = newl;Line(l31)=l31,l32};
l32 = newl;Line(l32)=l32,l33};
l33 = newl;Line(l33)=l33,l34};
l34 = newl;Line(l34)=l34,l31};
l35 = newl;Line(l35)=l21,l31};
l36 = newl;Line(l36)=l22,l32};
l37 = newl;Line(l37)=l23,l33};
l38 = newl;Line(l38)=l24,l34};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 4 // 
// // // // // // // // // // // // //

Transfinite Line {l31} = tf + 1;
Transfinite Line {l32} = tf + 1;
Transfinite Line {l33} = tf + 1;
Transfinite Line {l34} = tf;
Transfinite Line {l35} = 50;
Transfinite Line {l36} = 50;
Transfinite Line {l37} = 50;
Transfinite Line {l38} = 50;


// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 5 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p41 = newp;Point(p41) = {0, 0, p41*n, la};
p42 = newp;Point(p42 ) = {length, 0,p42*n, la};
p43 = newp;Point(p43) = {length, length,p43*n, la};
p44 = newp;Point(p44) = {0, length, p44*n, la};

// // Define Lines and Volume /// 

l41 = newl;Line(l41)=l41,l42};
l42 = newl;Line(l42)=l42,l43};
l43 = newl;Line(l43)=l43,l44};
l44 = newl;Line(l44)=l44,l41};
l45 = newl;Line(l45)=l31,l41};
l46 = newl;Line(l46)=l32,l42};
l47 = newl;Line(l47)=l33,l43};
l48 = newl;Line(l48)=l34,l44};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 5 // 
// // // // // // // // // // // // //

Transfinite Line {l41} = tf + 1;
Transfinite Line {l42} = tf + 1;
Transfinite Line {l43} = tf + 1;
Transfinite Line {l44} = tf;
Transfinite Line {l45} = 50;
Transfinite Line {l46} = 50;
Transfinite Line {l47} = 50;
Transfinite Line {l48} = 50;


// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 6 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p51 = newp;Point(p51) = {0, 0, p51*n, la};
p52 = newp;Point(p52 ) = {length, 0,p52*n, la};
p53 = newp;Point(p53) = {length, length,p53*n, la};
p54 = newp;Point(p54) = {0, length, p54*n, la};

// // Define Lines and Volume /// 

l51 = newl;Line(l51)=l51,l52};
l52 = newl;Line(l52)=l52,l53};
l53 = newl;Line(l53)=l53,l54};
l54 = newl;Line(l54)=l54,l51};
l55 = newl;Line(l55)=l41,l51};
l56 = newl;Line(l56)=l42,l52};
l57 = newl;Line(l57)=l43,l53};
l58 = newl;Line(l58)=l44,l54};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 6 // 
// // // // // // // // // // // // //

Transfinite Line {l51} = tf + 1;
Transfinite Line {l52} = tf + 1;
Transfinite Line {l53} = tf + 1;
Transfinite Line {l54} = tf;
Transfinite Line {l55} = 37.0 Using Progression 1.078;
Transfinite Line {l56} = 37.0 Using Progression 1.078;
Transfinite Line {l57} = 37.0 Using Progression 1.078;
Transfinite Line {l58} = 37.0 Using Progression 1.078;


// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 7 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p61 = newp;Point(p61) = {0, 0, p61*n, la};
p62 = newp;Point(p62 ) = {length, 0,p62*n, la};
p63 = newp;Point(p63) = {length, length,p63*n, la};
p64 = newp;Point(p64) = {0, length, p64*n, la};

// // Define Lines and Volume /// 

l61 = newl;Line(l61)=l61,l62};
l62 = newl;Line(l62)=l62,l63};
l63 = newl;Line(l63)=l63,l64};
l64 = newl;Line(l64)=l64,l61};
l65 = newl;Line(l65)=l51,l61};
l66 = newl;Line(l66)=l52,l62};
l67 = newl;Line(l67)=l53,l63};
l68 = newl;Line(l68)=l54,l64};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 7 // 
// // // // // // // // // // // // //

Transfinite Line {l61} = tf + 1;
Transfinite Line {l62} = tf + 1;
Transfinite Line {l63} = tf + 1;
Transfinite Line {l64} = tf;
Transfinite Line {l65} = 37.0 Using Progression 0.928;
Transfinite Line {l66} = 37.0 Using Progression 0.928;
Transfinite Line {l67} = 37.0 Using Progression 0.928;
Transfinite Line {l68} = 37.0 Using Progression 0.928;

