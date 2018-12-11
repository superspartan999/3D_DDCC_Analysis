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

Transfinite Line {l1} = tf+1;
Transfinite Line {l2} = tf+1;
Transfinite Line {l3} = tf+1;
Transfinite Line {l4} = tf+1;

Transfinite Surface {ps1} = {1, 2, 3, 4};
// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 2 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p11 = newp;Point(p11) = {0, 0, 20.*n, la};
p12 = newp;Point(p12 ) = {length, 0,20.*n, la};
p13 = newp;Point(p13) = {length, length,20.*n, la};
p14 = newp;Point(p14) = {0, length, 20.*n, la};

// // Define Lines and Volume /// 

l11 = newl;Line(l11)={p11,p12};
l12 = newl;Line(l12)={p12,p13};
l13 = newl;Line(l13)={p13,p14};
l14 = newl;Line(l14)={p14,p11};
l15 = newl;Line(l15)={p1,p11};
l16 = newl;Line(l16)={p2,p12};
l17 = newl;Line(l17)={p3,p13};
l18 = newl;Line(l18)={p4,p14};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 2 // 
// // // // // // // // // // // // //

Transfinite Line {l11} = tf+1;
Transfinite Line {l12} = tf+1;
Transfinite Line {l13} = tf+1;
Transfinite Line {l14} = tf+1;
Transfinite Line {l15} = 37.0 Using Progression 1.078;
Transfinite Line {l16} = 37.0 Using Progression 1.078;
Transfinite Line {l17} = 37.0 Using Progression 1.078;
Transfinite Line {l18} = 37.0 Using Progression 1.078;

ll11 = newll; Line Loop (ll11) = {l11,l12,l13,l14};
ll12 = newll; Line Loop (ll12) = {l1,l16,-l11,-l15};
ll13 = newll; Line Loop (ll13) = {l2,l17,-l12,-l16};
ll14 = newll; Line Loop (ll14) = {l3,l18,-l13,-l17};
ll15 = newll; Line Loop (ll15) = {l4,l15,-l14,-l18};

ps11 = news; Plane Surface (ps11) = {ll11};
ps12 = news; Plane Surface (ps12) = {ll12};
ps13 = news; Plane Surface (ps13) = {ll13};
ps14 = news; Plane Surface (ps14) = {ll14};
ps15 = news; Plane Surface (ps15) = {ll15};

Transfinite Surface {ps11} ={p11,p12,p13,p14};
Transfinite Surface {ps12} ={p1,p2,p12,p11};
Transfinite Surface {ps13} ={p2,p3,p13,p12};
Transfinite Surface {ps14} ={p4,p3,p13,p14};
Transfinite Surface {ps15} ={p1,p4,p14,p11};

Surface Loop (1) ={ps1,ps11,ps12,ps13,ps14,ps15};
Volume (1) = {1};
Transfinite Volume (1);

// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 3 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p21 = newp;Point(p21) = {0, 0, 40.*n, la};
p22 = newp;Point(p22 ) = {length, 0,40.*n, la};
p23 = newp;Point(p23) = {length, length,40.*n, la};
p24 = newp;Point(p24) = {0, length, 40.*n, la};

// // Define Lines and Volume /// 

l21 = newl;Line(l21)={p21,p22};
l22 = newl;Line(l22)={p22,p23};
l23 = newl;Line(l23)={p23,p24};
l24 = newl;Line(l24)={p24,p21};
l25 = newl;Line(l25)={p11,p21};
l26 = newl;Line(l26)={p12,p22};
l27 = newl;Line(l27)={p13,p23};
l28 = newl;Line(l28)={p14,p24};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 3 // 
// // // // // // // // // // // // //

Transfinite Line {l21} = tf+1;
Transfinite Line {l22} = tf+1;
Transfinite Line {l23} = tf+1;
Transfinite Line {l24} = tf+1;
Transfinite Line {l25} = 37.0 Using Progression 0.928;
Transfinite Line {l26} = 37.0 Using Progression 0.928;
Transfinite Line {l27} = 37.0 Using Progression 0.928;
Transfinite Line {l28} = 37.0 Using Progression 0.928;

ll21 = newll; Line Loop (ll21) = {l21,l22,l23,l24};
ll22 = newll; Line Loop (ll22) = {l11,l26,-l21,-l25};
ll23 = newll; Line Loop (ll23) = {l12,l27,-l22,-l26};
ll24 = newll; Line Loop (ll24) = {l13,l28,-l23,-l27};
ll25 = newll; Line Loop (ll25) = {l14,l25,-l24,-l28};

ps21 = news; Plane Surface (ps21) = {ll21};
ps22 = news; Plane Surface (ps22) = {ll22};
ps23 = news; Plane Surface (ps23) = {ll23};
ps24 = news; Plane Surface (ps24) = {ll24};
ps25 = news; Plane Surface (ps25) = {ll25};

Transfinite Surface {ps21} ={p21,p22,p23,p24};
Transfinite Surface {ps22} ={p11,p12,p22,p21};
Transfinite Surface {ps23} ={p12,p13,p23,p22};
Transfinite Surface {ps24} ={p14,p13,p23,p24};
Transfinite Surface {ps25} ={p11,p14,p24,p21};

Surface Loop (2) ={ps11,ps21,ps22,ps23,ps24,ps25};
Volume (2) = {2};
Transfinite Volume (2);

// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 4 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p31 = newp;Point(p31) = {0, 0, 45.*n, la};
p32 = newp;Point(p32 ) = {length, 0,45.*n, la};
p33 = newp;Point(p33) = {length, length,45.*n, la};
p34 = newp;Point(p34) = {0, length, 45.*n, la};

// // Define Lines and Volume /// 

l31 = newl;Line(l31)={p31,p32};
l32 = newl;Line(l32)={p32,p33};
l33 = newl;Line(l33)={p33,p34};
l34 = newl;Line(l34)={p34,p31};
l35 = newl;Line(l35)={p21,p31};
l36 = newl;Line(l36)={p22,p32};
l37 = newl;Line(l37)={p23,p33};
l38 = newl;Line(l38)={p24,p34};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 4 // 
// // // // // // // // // // // // //

Transfinite Line {l31} = tf+1;
Transfinite Line {l32} = tf+1;
Transfinite Line {l33} = tf+1;
Transfinite Line {l34} = tf+1;
Transfinite Line {l35} = 50;
Transfinite Line {l36} = 50;
Transfinite Line {l37} = 50;
Transfinite Line {l38} = 50;

ll31 = newll; Line Loop (ll31) = {l31,l32,l33,l34};
ll32 = newll; Line Loop (ll32) = {l21,l36,-l31,-l35};
ll33 = newll; Line Loop (ll33) = {l22,l37,-l32,-l36};
ll34 = newll; Line Loop (ll34) = {l23,l38,-l33,-l37};
ll35 = newll; Line Loop (ll35) = {l24,l35,-l34,-l38};

ps31 = news; Plane Surface (ps31) = {ll31};
ps32 = news; Plane Surface (ps32) = {ll32};
ps33 = news; Plane Surface (ps33) = {ll33};
ps34 = news; Plane Surface (ps34) = {ll34};
ps35 = news; Plane Surface (ps35) = {ll35};

Transfinite Surface {ps31} ={p31,p32,p33,p34};
Transfinite Surface {ps32} ={p21,p22,p32,p31};
Transfinite Surface {ps33} ={p22,p23,p33,p32};
Transfinite Surface {ps34} ={p24,p23,p33,p34};
Transfinite Surface {ps35} ={p21,p24,p34,p31};

Surface Loop (3) ={ps21,ps31,ps32,ps33,ps34,ps35};
Volume (3) = {3};
Transfinite Volume (3);

// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 5 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p41 = newp;Point(p41) = {0, 0, 50.*n, la};
p42 = newp;Point(p42 ) = {length, 0,50.*n, la};
p43 = newp;Point(p43) = {length, length,50.*n, la};
p44 = newp;Point(p44) = {0, length, 50.*n, la};

// // Define Lines and Volume /// 

l41 = newl;Line(l41)={p41,p42};
l42 = newl;Line(l42)={p42,p43};
l43 = newl;Line(l43)={p43,p44};
l44 = newl;Line(l44)={p44,p41};
l45 = newl;Line(l45)={p31,p41};
l46 = newl;Line(l46)={p32,p42};
l47 = newl;Line(l47)={p33,p43};
l48 = newl;Line(l48)={p34,p44};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 5 // 
// // // // // // // // // // // // //

Transfinite Line {l41} = tf+1;
Transfinite Line {l42} = tf+1;
Transfinite Line {l43} = tf+1;
Transfinite Line {l44} = tf+1;
Transfinite Line {l45} = 50;
Transfinite Line {l46} = 50;
Transfinite Line {l47} = 50;
Transfinite Line {l48} = 50;

ll41 = newll; Line Loop (ll41) = {l41,l42,l43,l44};
ll42 = newll; Line Loop (ll42) = {l31,l46,-l41,-l45};
ll43 = newll; Line Loop (ll43) = {l32,l47,-l42,-l46};
ll44 = newll; Line Loop (ll44) = {l33,l48,-l43,-l47};
ll45 = newll; Line Loop (ll45) = {l34,l45,-l44,-l48};

ps41 = news; Plane Surface (ps41) = {ll41};
ps42 = news; Plane Surface (ps42) = {ll42};
ps43 = news; Plane Surface (ps43) = {ll43};
ps44 = news; Plane Surface (ps44) = {ll44};
ps45 = news; Plane Surface (ps45) = {ll45};

Transfinite Surface {ps41} ={p41,p42,p43,p44};
Transfinite Surface {ps42} ={p31,p32,p42,p41};
Transfinite Surface {ps43} ={p32,p33,p43,p42};
Transfinite Surface {ps44} ={p34,p33,p43,p44};
Transfinite Surface {ps45} ={p31,p34,p44,p41};

Surface Loop (4) ={ps31,ps41,ps42,ps43,ps44,ps45};
Volume (4) = {4};
Transfinite Volume (4);

// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 6 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p51 = newp;Point(p51) = {0, 0, 70.*n, la};
p52 = newp;Point(p52 ) = {length, 0,70.*n, la};
p53 = newp;Point(p53) = {length, length,70.*n, la};
p54 = newp;Point(p54) = {0, length, 70.*n, la};

// // Define Lines and Volume /// 

l51 = newl;Line(l51)={p51,p52};
l52 = newl;Line(l52)={p52,p53};
l53 = newl;Line(l53)={p53,p54};
l54 = newl;Line(l54)={p54,p51};
l55 = newl;Line(l55)={p41,p51};
l56 = newl;Line(l56)={p42,p52};
l57 = newl;Line(l57)={p43,p53};
l58 = newl;Line(l58)={p44,p54};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 6 // 
// // // // // // // // // // // // //

Transfinite Line {l51} = tf+1;
Transfinite Line {l52} = tf+1;
Transfinite Line {l53} = tf+1;
Transfinite Line {l54} = tf+1;
Transfinite Line {l55} = 37.0 Using Progression 1.078;
Transfinite Line {l56} = 37.0 Using Progression 1.078;
Transfinite Line {l57} = 37.0 Using Progression 1.078;
Transfinite Line {l58} = 37.0 Using Progression 1.078;

ll51 = newll; Line Loop (ll51) = {l51,l52,l53,l54};
ll52 = newll; Line Loop (ll52) = {l41,l56,-l51,-l55};
ll53 = newll; Line Loop (ll53) = {l42,l57,-l52,-l56};
ll54 = newll; Line Loop (ll54) = {l43,l58,-l53,-l57};
ll55 = newll; Line Loop (ll55) = {l44,l55,-l54,-l58};

ps51 = news; Plane Surface (ps51) = {ll51};
ps52 = news; Plane Surface (ps52) = {ll52};
ps53 = news; Plane Surface (ps53) = {ll53};
ps54 = news; Plane Surface (ps54) = {ll54};
ps55 = news; Plane Surface (ps55) = {ll55};

Transfinite Surface {ps51} ={p51,p52,p53,p54};
Transfinite Surface {ps52} ={p41,p42,p52,p51};
Transfinite Surface {ps53} ={p42,p43,p53,p52};
Transfinite Surface {ps54} ={p44,p43,p53,p54};
Transfinite Surface {ps55} ={p41,p44,p54,p51};

Surface Loop (5) ={ps41,ps51,ps52,ps53,ps54,ps55};
Volume (5) = {5};
Transfinite Volume (5);

// // // // // // // // // // // // // // // // // // // // // / 
                    // ** ***NEW VOL 7 ***  // // // /
// // // // // // // // // // // // // // // // // // // // // /

// // Define Points // // // /

p61 = newp;Point(p61) = {0, 0, 90.*n, la};
p62 = newp;Point(p62 ) = {length, 0,90.*n, la};
p63 = newp;Point(p63) = {length, length,90.*n, la};
p64 = newp;Point(p64) = {0, length, 90.*n, la};

// // Define Lines and Volume /// 

l61 = newl;Line(l61)={p61,p62};
l62 = newl;Line(l62)={p62,p63};
l63 = newl;Line(l63)={p63,p64};
l64 = newl;Line(l64)={p64,p61};
l65 = newl;Line(l65)={p51,p61};
l66 = newl;Line(l66)={p52,p62};
l67 = newl;Line(l67)={p53,p63};
l68 = newl;Line(l68)={p54,p64};


// // // // // // // // // // // // //
 // Mesh Definition of Volume 7 // 
// // // // // // // // // // // // //

Transfinite Line {l61} = tf+1;
Transfinite Line {l62} = tf+1;
Transfinite Line {l63} = tf+1;
Transfinite Line {l64} = tf+1;
Transfinite Line {l65} = 37.0 Using Progression 0.928;
Transfinite Line {l66} = 37.0 Using Progression 0.928;
Transfinite Line {l67} = 37.0 Using Progression 0.928;
Transfinite Line {l68} = 37.0 Using Progression 0.928;

ll61 = newll; Line Loop (ll61) = {l61,l62,l63,l64};
ll62 = newll; Line Loop (ll62) = {l51,l66,-l61,-l65};
ll63 = newll; Line Loop (ll63) = {l52,l67,-l62,-l66};
ll64 = newll; Line Loop (ll64) = {l53,l68,-l63,-l67};
ll65 = newll; Line Loop (ll65) = {l54,l65,-l64,-l68};

ps61 = news; Plane Surface (ps61) = {ll61};
ps62 = news; Plane Surface (ps62) = {ll62};
ps63 = news; Plane Surface (ps63) = {ll63};
ps64 = news; Plane Surface (ps64) = {ll64};
ps65 = news; Plane Surface (ps65) = {ll65};

Transfinite Surface {ps61} ={p61,p62,p63,p64};
Transfinite Surface {ps62} ={p51,p52,p62,p61};
Transfinite Surface {ps63} ={p52,p53,p63,p62};
Transfinite Surface {ps64} ={p54,p53,p63,p64};
Transfinite Surface {ps65} ={p51,p54,p64,p61};

Surface Loop (6) ={ps51,ps61,ps62,ps63,ps64,ps65};
Volume (6) = {6};
Transfinite Volume (6);
