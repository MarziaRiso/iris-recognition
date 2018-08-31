#include "subject.h"

subject* subject_create()
{
	subject* eye = (subject*)calloc(1, sizeof(subject));
	if (eye == NULL) return NULL;

	eye->input = NULL;
	eye->mask = NULL;

	return eye;

}

void subject_free(subject* eye)
{
	eye->input.release();
	eye->mask.release();
	free(eye);
}