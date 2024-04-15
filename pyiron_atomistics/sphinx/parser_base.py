# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

__author__ = "Christoph Freysoldt"
__copyright__ = (
    "Copyright 2023, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Christoph Freysoldt"
__email__ = "freysoldt@mpie.de"
__status__ = "production"
__date__ = "Dec 8, 2023"

import re
from types import GeneratorType
import numpy as np


class KeywordTreeParser:
    """
    A base class to parse files block by block via keyword-triggered
    parsing routines organized in a tree. Parsing routines can
    add more levels of keyword->parse function maps. The file
    is read line by line on demand while parsing, so large files will not clobber
    memory.

    A parser routine can either return or yield (once!) to continue parsing. If it yields,
    the rest of the routine (after yield) will be executed when the next keyword
    of the current or a higher level is found.

    Every parser routine MUST remove the keyword from the lineview.

    A typical use will be

    class my_parser(KeywordTreeParser):
        def __init__(self,file)
            super ().__init__({
                    "key1" : self.parse_key1,
                    "key2" : self.parse_key2  })
            self.parse (file)

    """

    def __init__(self, keylevels=[]):
        if isinstance(keylevels, dict):
            keylevels = [keylevels]
        elif not isinstance(keylevels, list):
            raise TypeError
        self.keylevels = keylevels

    def parse(self, filename):
        """
        Parse a file using the current keylevels

        Args:
            filename ... the filename of the file to parse
        Returns: nothing
        """
        # --- initialization
        if len(self.keylevels) == 0:
            raise KeyError("No parsing functions available in keylevels")
        filehandle = open(filename)
        # the following properties only exist while parsing
        self.line = filehandle.__iter__()
        self.lineview = ""
        self.filename = filename
        self.lineno = 0
        self.line_from = 0
        while True:
            for keymap in self.keylevels:
                for key, func in keymap.items():
                    if key in self.lineview:
                        self._cleanup(keymap)
                        res = func()
                        if isinstance(res, GeneratorType):
                            res.send(None)
                            keymap["%finalize!"] = res
                        break
                else:
                    continue
                break
            else:
                try:
                    self.lineview = next(self.line)
                    self.lineno += 1
                    self.line_from = self.lineno
                except StopIteration:
                    break
        self._cleanup(self.keylevels[0])
        # call optional finalize routine
        self.finalize()
        close(filehandle)
        # clean up object properties that only exist during parsing
        del (self.filename, self.line, self.lineno, self.line_from, self.lineview)

    def finalize(self):
        pass

    def location(self):
        """Return the current parsing location (for error messages)"""
        return f"in file '{self.filename}' line" + (
            f" {self.lineno}"
            if self.lineno == self.line_from
            else f"s {self.line_from}..{self.lineno}"
        )

    def read_until(self, match):
        """
        Appends more lines from input until match is found

        Args:
           match ... (str) what to wait for
        Returns: nothing
        """
        while not match in self.lineview:
            self.lineview += next(self.line)
            self.lineno += 1
            self.line_from = self.line

    def extract_via_regex(self, regex):
        """
        Extracts and removes some text from current lineview

        Args:
           regex ... regular expression
        Returns:
           the extracted text
        """
        if isinstance(regex, str):
            regex = re.compile(regex, re.DOTALL)
        result = regex.search(self.lineview)
        if result is None:
            raise RuntimeError(
                f"Failed to extract '{regex.pattern}' "
                + self.location()
                + "\n"
                + self.lineview
            )
        self.lineview = regex.sub("", self.lineview, count=1)
        return result.group()

    def _cleanup(self, active):
        """
        (internal routine) remove levels below the current (active) level, and
         call (optional) final blocks up to the current level

        Args:
           active ... the currently active map
        Returns:
           the extracted text
        """

        def try_finalize(keymap):
            if "%finalize!" in keymap:
                try:
                    next(keymap["%finalize!"])
                except StopIteration:
                    pass
                del keymap["%finalize!"]

        # roll back keylevels until active level
        while self.keylevels[-1] is not active:
            try_finalize(self.keylevels[-1])
            del self.keylevels[-1]
        # and call optional finalize of currently active level
        try_finalize(active)

    def get_vector(self, key, txt):
        """
        (auxiliary function) Get a vector from 'key = [ ... ] ;'

        Args:
           key ... the key to look for
        Returns:
           one-dimensional vector containing the numbers
        """
        # get the relevant part between '=' and ';'
        vecstring = re.sub(".*" + key + r"\s*=\s*([^;]+);.*", r"\1", txt)
        if vecstring is None:
            raise RuntimeError(
                f"Cannot parse {key} from '{txt}' as vector " + self.location()
            )
        # remove special characters [] , ; =
        vecstring = re.sub(r"[][=,;$]", " ", vecstring)
        return np.fromstring(vecstring, sep=" ")

    def extract_var(self, key, startend="=;"):
        """
        Extract a block 'key = ... ;'

        If the end pattern is not found in lineview, more lines are read.

        Args:
            key      ... the keyword
            startend ... (optional) Override the = ; pair by two different patterns
        Returns:
            the extracted block
        """
        self.read_until(startend[1])
        return self.extract_via_regex(
            key + r"\s*" + startend[0] + r"\s*[^" + startend[1] + "]+" + startend[1]
        )
