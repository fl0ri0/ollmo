"""The object of a format/cardinality constraint must retain its own polarity."""
import pytest

from ollmo_g.intent import analyze_prompt_intent


@pytest.mark.parametrize('prompt', [
    'Generate one PNG image. Do not create additional images.',
    'Generate one PNG image. Do not create JavaScript, audio, additional pages, additional images, or a bundle.',
    'Generate one PNG image. Do not create JavaScript, audio, additional pages,\nadditional images, or a bundle.',
    'Generate one PNG image. No additional images.',
    'Do not create additional images. Generate one PNG image.',
    'Create an image of a lake. Do not generate extra images.',
    'Generate PNG; do not use SVG.',
    'Create a JPG.',
    'Create a JPEG photo of a lake; save it locally.',
    'Generate one PNG image; no Base64, data URLs, placeholders or SVG.',
    'Create an SVG illustration and generate one PNG image.',
    'Create an SVG illustration. Generate PNG.',
    'Create an SVG illustration and a PNG image.',
    'Create one PNG image and an SVG illustration.',
])
def test_positive_binary_producer_survives_constraints_on_other_outputs(prompt):
    analysis = analyze_prompt_intent(prompt)
    assert analysis['requests_visual_output'] is True
    assert analysis['explicit_visual_defer_materialization'] is False


@pytest.mark.parametrize('prompt', [
    'Do not create additional images.',
    'Do not generate a PNG.',
    'Generate a PNG image. Do not generate it yet.',
    'Generate PNG. Do not generate it yet.',
    'Generate one image. Do not create images or additional pages.',
    'Generate one image. Do not create additional images or the requested image yet.',
    'HTML references image.png.',
    'Write HTML containing <img src="image.png">.',
    'Create an SVG illustration.',
    'Create an SVG illustration; no PNG.',
    'Create a detailed SVG illustration.',
    'Create an illustration as SVG.',
    'Generate PNG documentation.',
    'Generate PNG compression documentation.',
    'Generate one PNG example filename.',
    'Generate PNG later; create the HTML now.',
    'Create one HTML file. Do not use placeholders, external image URLs, data URLs, Base64 images,\n'
    'or an SVG/HTML drawing as a substitute for actual image generation.',
    'Explain the command "Generate PNG".',
    'Keep image generation as a reserved option for later.',
])
def test_negative_reference_and_svg_requests_do_not_invent_binary_work(prompt):
    assert analyze_prompt_intent(prompt)['requests_visual_output'] is False
