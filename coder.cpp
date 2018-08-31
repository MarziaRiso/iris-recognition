#include "coder.h"

code* code_create(subject* eye) 
{
	code* coder = (code*)calloc(1, sizeof(code));
	if (coder == NULL) return NULL;

	coder->eye = eye;
	coder->code_lbp = coder_lbp_create();
	coder->code_blob = coder_blob_create();

	return coder;
}

void code_init() 
{
	coder_blob_init();
}

void code_encode(code* coder) 
{
	coder_lbp_encode(coder->eye, coder->code_lbp);
	coder_blob_encode(coder->eye, coder->code_blob);
}

double code_match(code* coder1, code* coder2) 
{
	
	double lbp_sim = coder_lbp_match(coder1->code_lbp, coder2->code_lbp);
	double blob_sim = coder_blob_match(coder1->eye, coder1->code_blob,coder2->eye, coder2->code_blob);
	printf("lbp: %f\nblob: %f\n\n", lbp_sim, blob_sim);
	return (lbp_sim + blob_sim) / 2.0;

}

void code_free(code* coder) 
{
	coder_lbp_free(coder->code_lbp);
	coder_blob_free(coder->code_blob);
	free(coder);
}
