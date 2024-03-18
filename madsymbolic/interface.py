#####################################################
#                                                   #
#  Source file of the alphaLoop MG5aMC plugin.      #
#                                                   #
#####################################################

from distutils.version import LooseVersion, StrictVersion
import madsymbolic.utils as utils
import traceback
import resource
import copy
import functools
import timeit
import math
import random
import re
import shutil
import logging
from argparse import ArgumentParser
import os
import sys
import copy
import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


root_path = os.path.dirname(os.path.realpath(__file__))   # nopep8
sys.path.insert(0, os.path.normpath(os.path.join(root_path, os.path.pardir, os.path.pardir, os.path.pardir)))   # nopep8

import madgraph.interface.master_interface as master_interface  # type: ignore  # nopep8
from madgraph import InvalidCmd, MadGraph5Error, MG5DIR, ReadWrite  # type: ignore   # nopep8
import madgraph  # type: ignore   # nopep8
import madgraph.interface.madgraph_interface as madgraph_interface  # type: ignore  # nopep8
import madgraph.interface.extended_cmd as cmd  # type: ignore  # nopep8
import madgraph.loop.loop_base_objects as loop_base_objects  # type: ignore  # nopep8
import madgraph.interface.loop_interface as loop_interface  # type: ignore  # nopep8
import madgraph.loop.loop_diagram_generation as loop_diagram_generation  # type: ignore  # nopep8

plugin_path = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger('madsymbolic.Interface')

pjoin = os.path.join
template_dir = pjoin(plugin_path, 'Templates')


class madSymbolicInterfaceError(MadGraph5Error):
    """ Error for the madSymbolic plugin """
    pass


class madSymbolicInvalidCmd(InvalidCmd):
    """ Invalid command issued to the madSymbolic interface. """
    pass


class madSymbolicInterface(master_interface.MasterCmd, madgraph_interface.CmdExtended):
    # class madSymbolicInterface(madgraph_interface.MadGraphCmd, madgraph_interface.CmdExtended):
    """ Interface for steering the madSymbolic output.
    We make it inherit from CmdShell so that launch_ext_prog does not attempt to start in WebMode."""

    _supported_FORM_output_formats = [None, 'raw', 'c', 'pySecDec']

    def __init__(self, *args, **opts):
        """ Define attributes of this class."""

        self.madsymbolic_options = {
            'verbosity': 1,
            'gammaloop_path': None
        }

        self.gammaloop_interface = None

        super(madSymbolicInterface, self).__init__(*args, **opts)

    def parse_set_option(self, args):
        """ Parsing arguments/options passed to the command set_madsymbolic option."""

        options = {}

        # First combine all value of the options (starting with '--') separated by a space
        opt_args = []
        new_args = []
        for arg in args:
            if arg.startswith('--'):
                opt_args.append(arg)
            elif len(opt_args) > 0:
                opt_args[-1] += ' %s' % arg
            else:
                new_args.append(arg)

        for arg in opt_args:
            try:
                key, value = arg.split('=')
            except:
                key, value = arg, None
            key = key[2:]

            # All options are declared valid in this contex
            options[key] = eval(str(value))

        return new_args, options

    @classmethod
    def verbose_yaml_dump(cls, data):
        return yaml.dump(data, Dumper=NoAliasDumper, default_flow_style=False, sort_keys=False)

    @classmethod
    def split_str_args(cls, str_args: str) -> list[str]:
        return str_args.split(' ') if str_args != '' else []

    def do_display_madsymbolic_option(self, line):
        """ Display madsymbolic options"""
        logger.info('%sGeneral madSymbolic options%s' %
                    (utils.bcolors.GREEN, utils.bcolors.ENDC))
        logger.info('%s-----------------------%s' %
                    (utils.bcolors.GREEN, utils.bcolors.ENDC))
        for opt in sorted(self.madsymbolic_options.keys()):
            logger.info('%-30s : %s' %
                        (opt, str(self.madsymbolic_options[opt])))

    def do_set_madsymbolic_option(self, line):
        """ Logic for setting madsymbolic options."""
        args = self.split_arg(line)
        args, options = self.parse_set_option(args)
        key, value = args[:2]

        if key == 'verbosity':
            try:
                verbosity = eval(value)
            except:
                raise madSymbolicInvalidCmd(
                    "Specified verbosity '%s' is not an integer." % value)
            self.madsymbolic_options[key] = verbosity
        elif key == 'gammaloop_path':
            if not os.path.isfile(os.path.join(value, '_gammaloop.so')):
                if not os.path.isfile(os.path.join(value, 'python', 'gammaloop', '_gammaloop.so')):
                    raise madSymbolicInvalidCmd(
                        "Specified path gammaloop path '%s' does not contain the gammaloop library `_gammaloop.so`." % value)
                else:
                    self.madsymbolic_options[key] = os.path.join(
                        value, 'python')
            else:
                self.madsymbolic_options[key] = os.path.normpath(
                    os.path.join(value, os.path.pardir))
        else:
            raise madSymbolicInvalidCmd(
                "Unrecognized MadSymbolic option: %s" % key)

    def do_force_loop_model(self, line):
        """Force the use of the loop model for the current model."""
        logger.info(
            "Forcing model to use Feynman gauge and support arbitrary loops.")
        self.do_set('gauge Feynman')
        self._curr_model = loop_base_objects.LoopModel(
            init_dict=self._curr_model)
        self._curr_model.set('perturbation_couplings',
                             list(self._curr_model.get_coupling_orders()))

    def fix_fermion_flow(self, vertices, edges, entry_edge_ids):

        while len(entry_edge_ids) > 0:
            e_id = entry_edge_ids.pop()
            # print('\n>> e_id = ', e_id, '\n')
            if not edges[e_id]['fermion_flow_assigned']:
                raise madSymbolicInterfaceError(
                    "Error #1 in Inconsistency fermion-flow fixing.")
            next_vertex_id = None
            # print("edges[e_id]['vertices'] = ", edges[e_id]['vertices'])
            # print("ff", (vertices[edges[e_id]['vertices'][0]]
            #       ['PDGs'], vertices[edges[e_id]['vertices'][1]]['PDGs']))
            match (vertices[edges[e_id]['vertices'][0]]['PDGs'], vertices[edges[e_id]['vertices'][1]]['PDGs']):
                case (None, None):
                    # This edge cannot be processed yet because both ends have untreated vertices.
                    # This typically happens when starting from a loop line that was not processed yet.
                    continue
                case (None, _):
                    next_vertex_id = edges[e_id]['vertices'][0]
                case (_, None):
                    next_vertex_id = edges[e_id]['vertices'][1]
                case (_, _):
                    next_vertex_id = None

            # print('next_vertex_id = ', next_vertex_id)
            if next_vertex_id is None:
                continue
            next_vertex = vertices[next_vertex_id]

            next_outgoing_fermion_type = 0
            next_edge_id = None
            # print("next_vertex['edge_ids'] = ", next_vertex['edge_ids'])
            vertex_can_be_processed = True
            for e_id in next_vertex['edge_ids']:
                # print("e_id, edges[e_id]['fermion_flow_assigned'], edges[e_id]= ",
                #       e_id, edges[e_id]['fermion_flow_assigned'], edges[e_id])
                if edges[e_id]['fermion_flow_assigned']:
                    part = self._curr_model.get_particle(
                        edges[e_id]['PDG'])
                    if part.is_fermion():
                        if edges[e_id]['vertices'][1] == next_vertex_id:  # incoming fermion
                            next_outgoing_fermion_type += (
                                1 if part['is_part'] else -1)
                        else:
                            next_outgoing_fermion_type -= (
                                1 if part['is_part'] else -1)
                    # print(next_outgoing_fermion_type)

                else:
                    if next_edge_id is not None:
                        # This vertex cannot be fixed yet because it has more than one unassigned edge
                        vertex_can_be_processed = False
                        break
                    next_edge_id = e_id
            if not vertex_can_be_processed:
                continue
            # print('---')
            # print('edges = ', edges)
            # print('vertices = ', vertices)
            # print('e_id = ', e_id)
            # print('edges[e_id] = ', edges[e_id])
            # print('next_vertex_id = ', next_vertex_id)
            # print('vertices[next_vertex_id] = ', vertices[next_vertex_id])
            # print('next_edge_id = ', next_edge_id)
            if next_edge_id is None:
                if len(next_vertex['edge_ids']) > 1 and next_outgoing_fermion_type != 0:
                    raise madSymbolicInterfaceError(
                        "Error #4 in Inconsistency fermion-flow fixing.")
                next_vertex['PDGs'] = [edges[e_id]['PDG']
                                       for e_id in next_vertex['edge_ids']]
                continue

            next_edge = edges[next_edge_id]
            next_edge_part = self._curr_model.get_particle(
                next_edge['PDG'])
            next_edge_part_type = 1
            if next_edge_part.is_fermion():
                if abs(next_outgoing_fermion_type) != 1:
                    raise madSymbolicInterfaceError(
                        "Error #5 in Inconsistency fermion-flow fixing.")
                if next_vertex_id == next_edge['vertices'][0]:
                    # print('OUTGOING! next_outgoing_fermion_type = ',next_outgoing_fermion_type)
                    next_edge_part_type = next_outgoing_fermion_type
                elif next_vertex_id == next_edge['vertices'][1]:
                    # print('INCOMING! next_outgoing_fermion_type = ',next_outgoing_fermion_type)
                    next_edge_part_type = -next_outgoing_fermion_type
                else:
                    raise madSymbolicInterfaceError(
                        "Error #6 in Inconsistency fermion-flow fixing.")
            else:
                if next_outgoing_fermion_type != 0:
                    raise madSymbolicInterfaceError(
                        "Error #7 in Inconsistency fermion-flow fixing.")
            if next_edge_part_type == 1:
                next_edge['PDG'] = self._curr_model.get_particle(
                    next_edge_part['pdg_code']).get_pdg_code()
            else:
                next_edge['PDG'] = self._curr_model.get_particle(
                    next_edge_part['pdg_code']).get_anti_pdg_code()
            # print("new pdg = ", next_edge['PDG'])
            next_edge['fermion_flow_assigned'] = True
            next_vertex['PDGs'] = [edges[e_id]['PDG']
                                   for e_id in next_vertex['edge_ids']]
            # print("new vertex pdgs = ", next_vertex['PDGs'])

            # Recurse
            self.fix_fermion_flow(vertices, edges, [next_edge_id,])

    def parse_amplitude(self, amplitude, diagram_class='diagrams'):

        process = amplitude.get('process')
        incoming_edges = {leg.get('number'): {
            'name': f"p{leg.get('number')}",
            'PDG': leg.get('id'),
            'momentum': f"p{leg.get('number')}",
            'type': 'in',
            'indices': tuple([]),
            'vertices': [leg.get('number'), None],  # updated later
            'fermion_flow_assigned': True
        } for leg in process.get('legs') if leg.get('state') == False}
        incoming_vertices = {leg.get('number'): {
            "PDGs": (leg.get('id'),),
            "momenta": (f"p{leg.get('number')}",),
            "indices": tuple([]),
            "vertex_id": -1,
            "edge_ids": [leg.get('number'),]
        } for leg in process.get('legs') if leg.get('state') == False}
        outgoing_edges = {leg.get('number'): {
            'name': f"p{leg.get('number')}",
            'PDG': leg.get('id'),
            'momentum': f"p{leg.get('number')}",
            'type': 'out',
            'indices': tuple([]),
            'vertices': [None, leg.get('number')],  # updated later
            'fermion_flow_assigned': True
        } for leg in process.get('legs') if leg.get('state') == True}
        outgoing_vertices = {leg.get('number'): {
            "PDGs": (leg.get('id'),),
            "momenta": (f"p{leg.get('number')}",),
            "indices": tuple([]),
            "vertex_id": -1,
            "edge_ids": [leg.get('number'),]
        } for leg in process.get('legs') if leg.get('state') == True}

        graphs = []
        is_a_loop_diagram = (diagram_class == 'loop_diagrams')
        for diagram in amplitude.get(diagram_class):
            if is_a_loop_diagram and isinstance(diagram, loop_base_objects.LoopDiagram) and diagram.get('type') <= 0:
                continue
            # if is_a_loop_diagram:
            #     print(diagram.nice_string(amplitude.get('structure_repository')))
            # else:
            #     print(diagram.nice_string())

            edges = copy.deepcopy(incoming_edges)
            edges.update(copy.deepcopy(outgoing_edges))
            vertices = copy.deepcopy(incoming_vertices)
            vertices.update(copy.deepcopy(outgoing_vertices))
            i_vertex = max(vertices.keys())
            i_edge = max(edges.keys())
            edge_numbers_map = {
                edge_number: edge_number for edge_number in edges.keys()}
            diag_vertices = []
            starting_loop_i_edge = None

            if is_a_loop_diagram:
                if diagram.get('tag'):
                    for i, tag_elem in enumerate(diagram.get('tag')):
                        for j, struct in enumerate(tag_elem[1]):
                            diag_vertices.extend(amplitude.get('structure_repository')[
                                struct].get('vertices'))
                first_loop_vertex_id = i_vertex + len(diag_vertices) + 1
                diag_vertices.extend(diagram.get('vertices')[:-1])
                last_loop_vertex_id = i_vertex + len(diag_vertices)

                i_edge += 1
                first_loop_leg = [l for l in diagram.get('vertices')[0].get('legs')[
                    :-1] if l['loop_line']]
                if len(first_loop_leg) != 1:
                    raise madSymbolicInterfaceError(
                        f"Expected exactly one incoming loop leg in the first vertex of the following loop diagram in amplitude {amplitude.get('process').shell_string()} : {diagram.nice_string()}")
                first_loop_leg = first_loop_leg[0]

                starting_loop_i_edge = i_edge
                loop_part = self._curr_model.get_particle(
                    first_loop_leg.get('id'))
                if first_loop_leg['state'] == False:  # incoming leg
                    new_leg_pdg = loop_part.get_pdg_code()
                else:
                    new_leg_pdg = loop_part.get_anti_pdg_code()
                edges[starting_loop_i_edge] = {
                    'name': f"k{starting_loop_i_edge}",
                    'PDG': new_leg_pdg,
                    'momentum': f"k{starting_loop_i_edge}",
                    'type': 'virtual',
                    'indices': tuple([]),
                    # updated later
                    'vertices': [last_loop_vertex_id, first_loop_vertex_id],
                    'fermion_flow_assigned': True
                }
                edge_numbers_map[first_loop_leg.get(
                    'number')] = starting_loop_i_edge

            else:
                diag_vertices.extend(diagram.get('vertices'))

            for i_vert, vertex in enumerate(diag_vertices):

                legs = vertex.get('legs')

                i_vertex += 1
                # Last amplitude vertex does not create new edges
                connected_edge_ids = [
                    edge_numbers_map[leg.get('number')] for leg in legs[:-1]]

                final_part = self._curr_model.get_particle(legs[-1].get('id'))
                if i_vert == len(diag_vertices)-1:
                    if is_a_loop_diagram:
                        connected_edge_ids.append(starting_loop_i_edge)
                    else:
                        connected_edge_ids.append(
                            edge_numbers_map[legs[-1].get('number')])
                else:
                    i_edge += 1
                    new_leg_pdg = final_part.get_pdg_code()
                    edge_numbers_map[legs[-1].get('number')] = i_edge
                    edges[i_edge] = {
                        'name': f"{'k' if ('loop_line' in legs[-1] and legs[-1]['loop_line']) else 'q'}{i_edge}",
                        'PDG': new_leg_pdg,
                        'momentum': None,
                        'type': 'virtual',
                        'indices': tuple([]),
                        'vertices': [i_vertex, None],  # updated later
                        'fermion_flow_assigned': False
                    }
                    connected_edge_ids.append(i_edge)

                # This is not the right way to assign the PDG, simply follow what the leg id gives.
                # pdgs = [self._curr_model.get_particle(leg.get('id')).get_pdg_code() if leg.get(
                #    'state') == False else self._curr_model.get_particle(leg.get('id')).get_anti_pdg_code() for leg in legs]

                vertices[i_vertex] = {
                    "PDGs": None,  # will be set later
                    "momenta": tuple([]),
                    "indices": tuple([]),
                    # We cannot correlate vertex ID with the position of the vertex in the gammaloop parsed model at this stage.
                    # This is however fine so long as we consider a model with vertices uambiguously defined by the particles content.
                    "vertex_id": -1,
                    "edge_ids": tuple(connected_edge_ids),
                }
                # Update edge vertices to point to the new vertex
                for edge_id in (connected_edge_ids if i_vert == len(diag_vertices)-1 else connected_edge_ids[:-1]):
                    if edges[edge_id]['vertices'][1] is None:
                        edges[edge_id]['vertices'][1] = i_vertex
                    elif edges[edge_id]['vertices'][0] is None:
                        edges[edge_id]['vertices'][0] = i_vertex

            entry_edge_ids = (list(incoming_edges.keys()) +
                              list(outgoing_edges.keys()) +
                              ([starting_loop_i_edge,] if starting_loop_i_edge is not None else []))

            self.fix_fermion_flow(vertices, edges, entry_edge_ids)
            for edge in edges.values():
                if any(e_id is None for e_id in edge['vertices']):
                    raise madSymbolicInterfaceError(
                        f"Some edge vertices are not set for the following diagram in amplitude {amplitude.get('process').shell_string()} : {diagram.nice_string()}")
                if not edge['fermion_flow_assigned']:
                    raise madSymbolicInterfaceError(
                        f"Some edge was not fermion-flow fixed the following diagram in amplitude {amplitude.get('process').shell_string()} : {diagram.nice_string()}")
                del edge['fermion_flow_assigned']
                edge['vertices'] = tuple(edge['vertices'])

            for vertex in vertices.values():
                if vertex['PDGs'] is None:
                    raise madSymbolicInterfaceError(
                        f"Some vertex was not fermion-flow fixed for the following diagram in amplitude {amplitude.get('process').shell_string()} : {diagram.nice_string()}")
                vertex['PDGs'] = tuple(vertex['PDGs'])
                vertex['edge_ids'] = tuple(vertex['edge_ids'])

            graphs.append({
                'edges': edges,
                'nodes': vertices,
                'overall_factor': diagram.get('multiplier') if is_a_loop_diagram else 1
            })

        return graphs

    # import_graphs command
    write_graphs_parser = ArgumentParser(prog='write_graphs')
    write_graphs_parser.add_argument('output_path', metavar='output_path', type=str,
                                     help='Output directory to write the diagrams to.')
    write_graphs_parser.add_argument('--format', '-f', type=str, default='yaml',
                                     choices=['yaml',], help='Format to write the graphs to.')
    write_graphs_parser.add_argument('--file_name', '-fn', type=str, default=None,
                                     help='base file name for diagrams file output. Default: from process name.')

    def do_write_graphs(self, line):
        """Write amplitude graphs to files for processing. """

        if line == 'help':
            self.write_graphs_parser.print_help()
            return
        args = self.write_graphs_parser.parse_args(
            madSymbolicInterface.split_str_args(line))

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        for amplitude in self._curr_amps:
            diag_classes = None
            if isinstance(amplitude, loop_diagram_generation.LoopAmplitude):
                diag_classes = {'born_diagrams': 'tree',
                                'loop_diagrams': 'loop'}
            else:
                diag_classes = {'diagrams': 'tree'}

            for diag_class, diag_class_name in diag_classes.items():
                graph_dicts = self.parse_amplitude(
                    amplitude, diagram_class=diag_class)
                if len(graph_dicts) > 0:
                    with open(os.path.normpath(os.path.join(args.output_path, f"{diag_class_name}_{('amplitude_'+amplitude.get('process').shell_string() if args.file_name is None else args.file_name)}.yaml")), 'w') as f:
                        f.write(madSymbolicInterface.verbose_yaml_dump(
                            {'graphs': graph_dicts}))

    def do_reset_gammaloop(self, line):

        try:
            if self.madsymbolic_options['gammaloop_path'] is not None and self.madsymbolic_options['gammaloop_path'] not in sys.path:
                sys.path.insert(0, self.madsymbolic_options['gammaloop_path'])
            import gammaloop  # type: ignore
        except ImportError:
            raise madSymbolicInvalidCmd(
                "gammaloop package does not appear to be installed. You can install it with `pip install gammaloop`.")

        self.gammaloop_interface = gammaloop.GammaLoop()

        model_path_with_restriction = self._curr_model.get(
            'modelpath+restriction')
        model_path = self._curr_model.get('modelpath')
        if os.path.basename(model_path) == model_path:
            model_path_with_restriction = os.path.join(
                MG5DIR, 'models', model_path_with_restriction)
            model_path = os.path.join(MG5DIR, 'models', model_path)

        self.gammaloop_interface.run(gammaloop.CommandList.from_string(
            f"import_model {model_path_with_restriction} --format ufo"))

    def do_gL(self, line):
        """Run a gammaloop function."""

        try:
            if self.madsymbolic_options['gammaloop_path'] is not None and self.madsymbolic_options['gammaloop_path'] not in sys.path:
                sys.path.insert(0, self.madsymbolic_options['gammaloop_path'])
            import gammaloop  # type: ignore
        except ImportError as e:
            raise madSymbolicInvalidCmd(
                "gammaloop package does not appear to be installed or its import failed. You can install it with `pip install gammaloop`. Error: %s" % str(e))

        if self.gammaloop_interface is None:
            self.do_reset_gammaloop('')

        logger.info("Running gammaloop command: %s%s%s" %
                    (utils.bcolors.GREEN, line, utils.bcolors.ENDC))
        self.gammaloop_interface.run(  # type: ignore
            gammaloop.CommandList.from_string(line))

    def get_madsymbolic_banner(self):
        """ Returns a string of alpha loop banner."""

        res = []
        res.append(("%s"+"="*80+"=%s") %
                   (utils.bcolors.GREEN, utils.bcolors.ENDC))
        res.append(("%s||"+" "*36+u'MadSymbolic'+" "*36+"||%s") %
                   (utils.bcolors.GREEN, utils.bcolors.ENDC))
        res.append(("%s"+"="*80+"=%s") %
                   (utils.bcolors.GREEN, utils.bcolors.ENDC))
        return '\n'.join(res)

    # command to change the prompt
    def preloop(self, *args, **opts):
        """only change the prompt after calling  the mother preloop command"""

        # The colored prompt screws up the terminal for some reason.
        # self.prompt = '\033[92mGGVV > \033[0m'
        self.prompt = "MadSymbolic > "

        logger.info("\n\n%s\n" % self.get_madsymbolic_banner())

        # By default, load the UFO Standard Model
        logger.info("Loading default model for MadSymbolic: sm-full")
        self.exec_cmd('import model sm-full', printcmd=False, precmd=True)

        # preloop mother
        madgraph_interface.CmdExtended.preloop(self)
