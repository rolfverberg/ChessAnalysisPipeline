#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : reader.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: FOXDEN readers
"""

# System modules
import json

# Third party modules
import requests

# Local modules
from CHAP.foxden.utils import HttpRequest
from CHAP.reader import Reader


class FoxdenDataDiscoveryReader(Reader):
    """Reader for the FOXDEN Data Discovery service."""
    def read(self, config):
        """Read records from the FOXDEN Data Discovery service based on
        did or an arbitrary query.

        :param config: FOXDEN HTTP request configuration.
        :type config: CHAP.foxden.models.FoxdenRequestConfig
        :return: Discovered data records.
        :rtype: list
        """
        # Load and validate the FoxdenRequestConfig configuration
        config = self.get_config(
            config=config, schema='foxden.models.FoxdenRequestConfig')
        self.logger.debug(f'config: {config}')

        # Submit HTTP request and return response
        rurl = f'{config.url}/search'
        request = {'client': 'CHAP-FoxdenDataDiscoveryReader'}
        if config.did is None:
            if config.query is None:
                query = '{}'
            else:
                query = config.query
            request['service_query'] = {'query': query, 'limit': config.limit}
        else:
            if config.limit is not None:
                self.logger.warning(
                    f'Ignoring parameter "limit" ({config.limit}), '
                    'when "did" is specified')
            if config.query is not None:
                self.logger.warning(
                    f'Ignoring parameter "query" ({config.query}), '
                    'when "did" is specified')
            request['service_query'] = {'query': f'did:{config.did}'}
        payload = json.dumps(request)
        self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method='POST', scope='read')
        if config.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            result = json.loads(response.text)['results']['records']
        else:
            self.logger.warning(f'HTTP error code {response.status_code}')
            result = []
        self.logger.debug(f'Returning {len(result)} records')
        return result


class FoxdenMetadataReader(Reader):
    """Reader for the FOXDEN Metadata service."""
    def read(self, config):
        """Read records from the FOXDEN Metadata service based on did
        or an arbitrary query.

        :param config: FOXDEN HTTP request configuration.
        :type config: CHAP.foxden.models.FoxdenRequestConfig
        :return: Metadata records.
        :rtype: list
        """
        # Load and validate the FoxdenRequestConfig configuration
        config = self.get_config(
            config=config, schema='foxden.models.FoxdenRequestConfig')
        self.logger.debug(f'config: {config}')

        # Submit HTTP request and return response
        rurl = f'{config.url}/search'
        request = {'client': 'CHAP-FoxdenMetadataReader'}
        if config.did is None:
            if config.query is None:
                query = '{}'
            else:
                query = config.query
            request['service_query'] = {'query': query}
        else:
            if config.query is not None:
                self.logger.warning(
                    f'Ignoring parameter "query" ({config.query}), '
                    'when "did" is specified')
            request['service_query'] = {'query': f'did:{config.did}'}
        payload = json.dumps(request)
        self.logger.info(f'method=POST url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method='POST', scope='read')
        if config.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            result = json.loads(response.text)
        else:
            self.logger.warning(f'HTTP error code {response.status_code}')
            result = []
        self.logger.debug(f'Returning {len(result)} records')
        return result


class FoxdenProvenanceReader(Reader):
    """Reader for FOXDEN Provenance data from a specific FOXDEN
    Provenance service.
    """
    def read(self, config):
        """Read records from the FOXDEN Provenance service based on did
        or an arbitrary query.

        :param config: FOXDEN HTTP request configuration.
        :type config: CHAP.foxden.models.FoxdenRequestConfig
        :return: Provenance input and output file records.
        :rtype: list
        """
        # Load and validate the FoxdenRequestConfig configuration
        config = self.get_config(
            config=config, schema='foxden.models.FoxdenRequestConfig')
        self.logger.debug(f'config: {config}')

        # Submit HTTP request and return response
        rurl = f'{config.url}/files?did={config.did}'
        request = {'client': 'CHAP-FoxdenProvenanceReader'}
        if config.did is None:
            if config.query is None:
                query = '{}'
            else:
                query = config.query
            request['service_query'] = {'query': query, 'limit': config.limit}
        else:
            if config.limit is not None:
                self.logger.warning(
                    f'Ignoring parameter "limit" ({config.limit}), '
                    'when "did" is specified')
            if config.query is not None:
                self.logger.warning(
                    f'Ignoring parameter "query" ({config.query}), '
                    'when "did" is specified')
            request['service_query'] = {'query': f'did:{config.did}'}
        payload = json.dumps(request)
        self.logger.info(f'method=GET url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method='GET', scope='read')
        exit(f'\n\nresponse.text:\n{response.text}')
        if config.verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            result = [{'name': v['name'], 'file_type': v['file_type']}
                      for v in json.loads(response.text)]
        else:
            self.logger.warning(f'HTTP error code {response.status_code}')
            result = []
        self.logger.debug(f'Returning {len(result)} records')
        return result


class FoxdenSpecScansReader(Reader):
    """Reader for FOXDEN SpecScans data from a specific FOXDEN
    SpecScans service.
    """
    def read(
            self, url, data, did='', query='', spec=None, method='POST', #'GET',
            verbose=False):
#TODO FIX
        """Read and return data from a specific FOXDEN SpecScans
        service.

        :param url: URL of service.
        :type url: str
        :param data: Input data.
        :type data: list[PipelineData]
        :param did: FOXDEN dataset identifier (did).
        :type did: string, optional
        :param query: FOXDEN query.
        :type query: string, optional
        :param spec: FOXDEN spec.
        :type spec: dictionary, optional
        :param method: HTTP method to use, `'POST'` for creation or
            `'PUT'` for update, defaults to `'POST'`.
        :type method: str, optional
        :param verbose: Verbose output flag, defaults to `False`.
        :type verbose: bool, optional
        :return: Contents of the input data.
        :rtype: object
        """
        self.logger.info(
            f'Executing "process" with url={url} data={data} did={did}')
        rurl = f'{url}/search'
        request = {'client': 'CHAP-FoxdenSpecScansReader', 'service_query': {}}
        if did:
            request['service_query'].update({'spec': {'did': did}})
        if query:
            request['service_query'].update({'query': query})
        if spec:
            request['service_query'].update({'spec': spec})
        payload = json.dumps(request)
        if verbose:
            self.logger.info(f'method={method} url={rurl} payload={payload}')
        response = HttpRequest(rurl, payload, method=method)
        if verbose:
            self.logger.info(
                f'code={response.status_code} data={response.text}')
        if response.status_code == 200:
            data = json.loads(response.text)
        else:
            data = []
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.reader import main

    main()
