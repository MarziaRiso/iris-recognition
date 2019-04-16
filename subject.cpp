#include "subject.h"

/**
* Metodo che crea e inializza un subject
* @return puntatore a Subject
**/
subject* subject_create()
{
	subject* eye = (subject*)calloc(1, sizeof(subject));
	if (eye == NULL) return NULL;

	eye->input = NULL;
	eye->mask = NULL;

	return eye;

}


/**
* Metodo che rilascia le immagini di un subject e il suo spazio di memoria
* @param eye puntatore a subject
**/
void subject_free(subject* eye)
{
	eye->input.release();
	eye->mask.release();
	free(eye);
}