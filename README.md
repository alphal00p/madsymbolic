# MadSymbolic
*MadSymbolic* (or *MadSym* in short), is [*MadGraph5_aMC@NLO*](https://launchpad.net/mg5amcnlo) (*MG5aMC*) plugin designed to help export *MG5aMC* tree and loop amplitudes to [*gammaLoop*](https://github.com/alphal00p/gammaloop) where it can be further manipulated.

In particular, the amplitude expressions can be processed symbolically using [*gammaLoop*](https://github.com/alphal00p/gammaloop) and the computer algebra system [`Symbolica`](https://symbolica.io/).

## Installation

*MadSymbolic* has two core dependencies:

* [*MadGraph5_aMC@NLO*](https://launchpad.net/mg5amcnlo).

* [*gammaLoop*](https://github.com/alphal00p/gammaloop) which you can install from source or with `pip install gammaloop`. 

*PS: Note that for now, you need to work using the [*MadSymbolic* branch](https://github.com/alphal00p/gammaloop/tree/madsymbolic) of *gammaLoop**.

Then you can install *madsymbolic* by placing the git clone into the PLUGIN directory of your *MG5aMC* installation.

## Usage

The *MadSymbolic* plugin offers additional commands to the *MG5aMC* command line interface, which you can access by running `mg5_aMC` as follows:

```
./bin/mg5_aMC --mode=madsymbolic
```

## Commands available

The following new commands become available when running *MG5aMC* with the *MadSymbolic* plugin:

* `set_madsymbolic_option <OPTION_NAME> <OPTION_VALUE>`

    Specifies a particular option for the *MadSymbolic* plugin.
    
    e.g: `set_madsymbolic_option gammaloop_path /path/to/gammaloop/directory`

* `force_loop_model`

    Because *MadSymbolic* only considers bare symbolic amplitudes, it is useful to make it possible for *MG5aMC* to generate loop amplitudes from any [*UFO*](https://arxiv.org/abs/2304.09883) model, even those without the necessary additional Feynman rules encoding renormalisation and so-called `R2` counterterms. 
    
    This command thus forces *MG5aMC* to allow the generation of loop amplitudes irrespectively of the nature of the [*UFO*](https://arxiv.org/abs/2304.09883) model already imported.

* `write_graphs <OUTPUT_DIR_NAME> <OPTIONS>`

    This command is meant to be ran *after* having run a `generate` or `add process` command of *MG5aMC* but *before* running its `output` command. You can specify the option `--help` to get more information on the available options.
    The graphs contained in all tree and loop amplitudes generated by *MG5aMC* will be written to the specified directory, in a `YAML` format that can be read by *gammaLoop*. The set of graphs for the amplitude of each individual partonic channel will be written to separate files.

    e.g.: `write_graphs test_output --format yaml --file_name test`

* `gL <GAMMALOOP_COMMAND>`

    This command `gL` acts as a prefix for running *any* *gammaLoop* command. It will typically be used to further process the graphs written by `write_graphs` and to perform symbolic manipulations on the amplitude expressions.
    Note that *MadSymbolic* already automatically makes sure that *gammaLoop* has loaded the same [*UFO*](https://arxiv.org/abs/2304.09883) model that is currently active in *MG5aMC*.

    e.g.: `gL help`

## Example

Here is a simple example run of *MadSymbolic* with *MG5aMC* to generate all diagrams making up the tree and loop amplitudes of the process $q \bar{q} \to d \bar{d}g$. You can write a card `test.madsym` with the following content:

```bash
# Only necessary if you the gammaloop Python library is not in your PYTHONPATH
set_madsymbolic_option gammaloop_path PATH/TO/YOUR/GAMMALOOP/INSTALLATION
import model sm-no_widths
force_loop_model
define q = u u~ d d~
generate q q > d d~ g [virt=QCD]
write_graphs test_madsymbolic_output --format yaml
gL import_graphs test_madsymbolic_output/loop_amplitude_0_ddx_ddxg.yaml --format yaml
gL output test_gammaloop_output -mr -num -nf mathematica
```

and run it with `./bin/mg5_aMC --mode=madsymbolic test.madsym`.
You will find the graphs processed by *gammaLoop* in the directory `test_gammaloop_output`.
For example you can see the diagram renderings with:

```bash
cd test_gammaloop_output/sources/amplitudes/loop_amplitude_0_ddx_ddxg/drawings
make -j 16
open feynman_diagrams.pdf
```

And the numerator expressions in `test_gammaloop_output/sources/amplitudes/loop_amplitude_0_ddx_ddxg/numerator`.