# Makefile for genreating the ESM example program 
#
# $Id: Makefile,v 1.6 2005/04/27 17:18:00 sbenhima Exp $

CC = gcc

CCFLAGS = -O3 -funroll-loops
#CCFLAGS += -Wall 

INCDIR = -I./inc

SRCDIR = src

BINDIR = bin

RESDIR = res

LIBDIR = lib

RESDIR = res

LIBS = ESMlib.a

PROG = ESMexample 

STDLIB = -lm

LIBS := $(addprefix $(LIBDIR)/,$(LIBS))

all: $(addprefix $(BINDIR)/,$(PROG))

$(BINDIR)/%: $(SRCDIR)/%.c $(LIBS) 
	@if test ! -d $(BINDIR); then mkdir $(BINDIR); fi
	@if test ! -d $(RESDIR); then mkdir $(RESDIR); fi
	$(CC) $(CCFLAGS) -o $@ $(INCDIR) $< $(LIBS) $(STDLIB)

clean: 
	rm -rf $(BINDIR) $(RESDIR)
