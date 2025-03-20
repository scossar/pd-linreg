#include "m_pd.h"

static t_class *linreg_class;

typedef struct _linreg {
  t_object x_obj;

  t_int nx;
  t_int m;

  t_float alpha;

  t_float *x;
  t_float *y;
  t_float *w;
  t_float b;

  t_outlet *prediction_outlet;
} t_linreg;


void *linreg_new(t_symbol *s, int argc, t_atom *argv)
{
  t_linreg *x = (t_linreg *)pd_new(linreg_class);

  // Default values
  x->nx = 1;
  x->m = 1;
  x->alpha = 0.01;

  // Parse args (nx, m, alpha)
  if (argc >= 1) x->nx = atom_getint(argv);
  if (argc >= 2) x->m = atom_getint(argv+1);
  if (argc >= 3) x->alpha = atom_getfloat(argv+2);

  // Allocate memory
  x->x = (t_float *)getbytes(x->nx * x->m * sizeof(t_float));
  x->y = (t_float *)getbytes(x->m * sizeof(t_float));
  x->w = (t_float *)getbytes(x->nx * sizeof(t_float));

  // Initialize weights to zero
  for (int i = 0; i < x->nx; i++) {
    x->w[i] = 0.0;
  }

  x->b = 0.0;

  x->prediction_outlet = outlet_new(&x->x_obj, &s_list);

  return (void *)x;
}

void linreg_free(t_linreg *x)
{
  freebytes(x->x, x->nx * x->m * sizeof(t_float));
  freebytes(x->y, x->m * sizeof(t_float));
  freebytes(x->w, x->nx * sizeof(t_float));
}

void linreg_set_x(t_linreg *x, t_symbol *s, int argc, t_atom *argv)
{
  if (argc != x->nx * x->m) {
    pd_error(x, "linreg: expected %ld values for X (nx=%ld, m=%ld)",
             x->nx * x->m, x->nx, x->m);
    return;
  }

  for (int i = 0; i < argc; i++) {
    x->x[i] = atom_getfloat(argv + i);
  }
}

void linreg_set_y(t_linreg *x, t_symbol *s, int argc, t_atom *argv)
{
  if (argc != x->m) {
    pd_error(x, "linreg: expected %ld values for Y (m=%ld)", x->m, x->m);
    return;
  }

  for (int i = 0; i < argc; i++) {
    x->y[i] = atom_getfloat(argv + i);
  }
}

void linreg_forward(t_linreg *x, t_float *predictions)
{
  for (int i = 0; i < x->m; i++) {
    predictions[i] = x->b; // start with bias

    // add weighted sum of features
    for (int j = 0; j < x->nx; j++) {
      predictions[i] += x->w[j] * x->x[j * x->m + i];
    }
  }
}

void linreg_backward(t_linreg *x, t_float *predictions)
{
  t_float dw[x->nx];
  t_float db = 0.0;

  for (int j = 0; j < x->nx; j++) {
    dw[j] = 0.0;
  }

  for (int i = 0; i < x->m; i++) {
    t_float error = predictions[i] - x->y[i];

    for (int j = 0; j < x->nx; j++) {
      dw[j] += error * x->x[j * x->m + i];
    }
    db += error;
  }

  // scale gradients by m
  for (int j = 0; j < x->nx; j++) {
    dw[j] /= x->m;
  }
  db /= x->m;

  // update parameters
  for (int j = 0; j < x->nx; j++) {
    x->w[j] -= x->alpha * dw[j];
  }
  x->b -= x->alpha * db;
}

void linreg_bang(t_linreg *x)
{
  t_float predictions[x->m];
  t_atom pred_atoms[x->m];

  linreg_forward(x, predictions);

  // convert preditions to atoms for output
  for (int i = 0; i < x->m; i++) {
    SETFLOAT(&pred_atoms[i], predictions[i]);
  }

  outlet_list(x->prediction_outlet, &s_list, x->m, pred_atoms);

  linreg_backward(x, predictions);
}

void linreg_setup(void)
{
  linreg_class = class_new(gensym("linreg"),
                           (t_newmethod)linreg_new,
                           (t_method)linreg_free,
                           sizeof(t_linreg),
                           CLASS_DEFAULT,
                           A_GIMME,
                           0);

  class_addbang(linreg_class, linreg_bang);
  class_addmethod(linreg_class, (t_method)linreg_set_x, gensym("x"), A_GIMME, 0);
  class_addmethod(linreg_class, (t_method)linreg_set_y, gensym("y"), A_GIMME, 0);
}


